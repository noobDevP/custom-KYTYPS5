#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace Common {

class ThreadPool {
public:
	explicit ThreadPool(unsigned int threads = std::thread::hardware_concurrency());
	~ThreadPool();

	ThreadPool(const ThreadPool&)            = delete;
	ThreadPool& operator=(const ThreadPool&) = delete;

	void Enqueue(std::function<void()> fn);
	bool TryExecuteOne();

	[[nodiscard]] unsigned int ThreadCount() const noexcept {
		return static_cast<unsigned int>(workers_.size());
	}

	// Splits [0, count) across the worker pool and blocks until all chunks complete.
	template <typename Func>
	void ParallelFor(size_t count, Func&& fn, size_t min_items_per_task = 64) {
		if (count == 0) {
			return;
		}
		const size_t num_threads = workers_.empty() ? 1 : workers_.size();
		const size_t max_tasks   = (count + min_items_per_task - 1) / min_items_per_task;
		const size_t num_tasks   = std::min(num_threads, max_tasks);
		if (num_tasks <= 1) {
			for (size_t i = 0; i < count; ++i) {
				fn(i);
			}
			return;
		}

		const size_t        chunk_size = (count + num_tasks - 1) / num_tasks;
		std::atomic<size_t> next_chunk {0};
		std::atomic<size_t> remaining {num_tasks - 1};
		std::mutex          done_mutex;
		std::condition_variable done_cv;

		const auto run_chunks = [&]() {
			while (true) {
				const size_t chunk = next_chunk.fetch_add(1, std::memory_order_relaxed);
				if (chunk >= num_tasks) {
					break;
				}
				const size_t begin = chunk * chunk_size;
				const size_t end   = std::min(count, begin + chunk_size);
				for (size_t i = begin; i < end; ++i) {
					fn(i);
				}
			}
		};

		for (size_t task_idx = 1; task_idx < num_tasks; ++task_idx) {
			Enqueue([&run_chunks, &remaining, &done_mutex, &done_cv]() {
				run_chunks();
				if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
					std::lock_guard<std::mutex> lock(done_mutex);
					done_cv.notify_one();
				}
			});
		}

		// Calling thread actively steals chunks and drains queued tasks so nested ParallelFor never deadlocks.
		run_chunks();

		while (remaining.load(std::memory_order_acquire) != 0) {
			if (!TryExecuteOne()) {
				std::unique_lock<std::mutex> lock(done_mutex);
				if (remaining.load(std::memory_order_acquire) != 0) {
					done_cv.wait_for(lock, std::chrono::microseconds(100), [&remaining]() {
						return remaining.load(std::memory_order_acquire) == 0;
					});
				}
			}
		}
	}

private:
	void WorkerLoop();

	std::vector<std::thread>          workers_;
	std::queue<std::function<void()>> tasks_;
	std::mutex                        mutex_;
	std::condition_variable           cv_;
	std::atomic<bool>                 stop_ {false};
};

ThreadPool& GetThreadPool();

} // namespace Common

using Common::GetThreadPool;
using Common::ThreadPool;
