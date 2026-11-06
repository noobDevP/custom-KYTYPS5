#ifndef KYTY_COMMON_PARALLELFOR_H_
#define KYTY_COMMON_PARALLELFOR_H_

#include "common/threadPool.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>

namespace Common {

// Half the hardware threads, at least one.
inline size_t ParallelThreads() {
	return std::max(1u, std::thread::hardware_concurrency() / 2);
}

// Calls job(i) for every i below count on up to ParallelThreads() threads, the caller included.
template <typename Job>
void ParallelFor(size_t count, const Job& job) {
	if (count == 0) {
		return;
	}
	const size_t workers = std::min(count, ParallelThreads());
	if (workers <= 1) {
		for (size_t i = 0; i < count; ++i) {
			job(i);
		}
		return;
	}

	std::atomic_size_t      next      = 0;
	std::atomic_size_t      remaining = workers - 1;
	std::mutex              done_mtx;
	std::condition_variable done_cv;

	const auto work = [&] {
		for (auto i = next.fetch_add(1, std::memory_order_relaxed); i < count;
		     i      = next.fetch_add(1, std::memory_order_relaxed)) {
			job(i);
		}
	};

	auto& pool = GetThreadPool();
	for (size_t i = 1; i < workers; ++i) {
		pool.Enqueue([&] {
			work();
			if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
				std::lock_guard<std::mutex> lock(done_mtx);
				done_cv.notify_one();
			}
		});
	}

	work();

	if (remaining.load(std::memory_order_acquire) != 0) {
		std::unique_lock<std::mutex> lock(done_mtx);
		done_cv.wait(lock, [&] { return remaining.load(std::memory_order_acquire) == 0; });
	}
}

} // namespace Common

#endif // KYTY_COMMON_PARALLELFOR_H_
