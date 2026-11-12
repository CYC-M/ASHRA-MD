#pragma once

// Zero-dependency persistent C++17 worker pool for deterministic multi-core execution.
// Works identically on Linux EPYC containers (without requiring libgomp) and macOS Apple M4
// (without requiring Homebrew libomp), with zero thread creation overhead inside MD steps.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <algorithm>

namespace ashmd {

class ParallelPool {
public:
    explicit ParallelPool(int32_t num_threads = 1) {
        n_threads_ = std::max(1, num_threads);
        if (n_threads_ > 1) {
            workers_.reserve(n_threads_ - 1);
            for (int32_t t = 1; t < n_threads_; ++t) {
                workers_.emplace_back([this, t]() { worker_loop(t); });
            }
        }
    }

    ~ParallelPool() {
        if (n_threads_ > 1) {
            {
                std::lock_guard<std::mutex> lk(mtx_);
                stop_ = true;
                ++epoch_;
            }
            cv_start_.notify_all();
            for (auto& w : workers_) {
                if (w.joinable()) w.join();
            }
        }
    }

    ParallelPool(const ParallelPool&) = delete;
    ParallelPool& operator=(const ParallelPool&) = delete;

    int32_t num_threads() const noexcept { return n_threads_; }

    // Execute fn(thread_id, begin, end) across [0, count) with dynamic chunking for load balance
    template <typename Func>
    void parallel_for(int32_t count, int32_t chunk_size, Func&& fn) {
        if (count <= 0) return;
        if (n_threads_ <= 1 || count <= chunk_size) {
            fn(0, 0, count);
            return;
        }
        next_idx_.store(0, std::memory_order_relaxed);
        int32_t step = std::max(1, chunk_size);

        task_ = [this, count, step, &fn](int32_t tid) {
            while (true) {
                int32_t beg = next_idx_.fetch_add(step, std::memory_order_relaxed);
                if (beg >= count) break;
                int32_t end = std::min(count, beg + step);
                fn(tid, beg, end);
            }
        };

        {
            std::lock_guard<std::mutex> lk(mtx_);
            pending_ = n_threads_ - 1;
            ++epoch_;
        }
        cv_start_.notify_all();

        // Master thread (tid = 0) participates directly
        task_(0);

        // Wait for workers
        std::unique_lock<std::mutex> lk(mtx_);
        cv_done_.wait(lk, [this]() { return pending_ == 0; });
    }

    // Execute 1 independent task per thread (e.g., 4 coupled replicas on 4 Instabox vCPUs!)
    template <typename Func>
    void parallel_ntasks(int32_t n_tasks, Func&& fn) {
        if (n_tasks <= 0) return;
        if (n_threads_ <= 1 || n_tasks == 1) {
            for (int32_t i = 0; i < n_tasks; ++i) fn(0, i);
            return;
        }
        parallel_for(n_tasks, 1, [&](int32_t tid, int32_t beg, int32_t end) {
            for (int32_t i = beg; i < end; ++i) {
                fn(tid, i);
            }
        });
    }

private:
    void worker_loop(int32_t tid) {
        uint64_t seen_epoch = 0;
        while (true) {
            std::function<void(int32_t)> local_task;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_start_.wait(lk, [this, &seen_epoch]() {
                    return stop_ || epoch_ != seen_epoch;
                });
                if (stop_) return;
                seen_epoch = epoch_;
                local_task = task_;
            }
            local_task(tid);
            {
                std::lock_guard<std::mutex> lk(mtx_);
                if (--pending_ == 0) {
                    cv_done_.notify_one();
                }
            }
        }
    }

    int32_t n_threads_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mtx_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    std::function<void(int32_t)> task_;
    std::atomic<int32_t> next_idx_{0};
    uint64_t epoch_ = 0;
    int32_t pending_ = 0;
    bool stop_ = false;
};

} // namespace ashmd
