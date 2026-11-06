#include "common/threadPool.h"

#include <algorithm>

namespace Common {

ThreadPool::ThreadPool(unsigned int threads) {
	if (threads == 0) {
		threads = 1;
	}
	workers_.reserve(threads);
	for (unsigned int i = 0; i < threads; ++i) {
		workers_.emplace_back(&ThreadPool::WorkerLoop, this);
	}
}

ThreadPool::~ThreadPool() {
	{
		std::unique_lock<std::mutex> lock(mutex_);
		stop_ = true;
	}
	cv_.notify_all();
	for (auto& t: workers_) {
		if (t.joinable()) {
			t.join();
		}
	}
}

void ThreadPool::Enqueue(std::function<void()> fn) {
	{
		std::unique_lock<std::mutex> lock(mutex_);
		tasks_.push(std::move(fn));
	}
	cv_.notify_one();
}

bool ThreadPool::TryExecuteOne() {
	std::function<void()> task;
	{
		std::unique_lock<std::mutex> lock(mutex_);
		if (tasks_.empty()) {
			return false;
		}
		task = std::move(tasks_.front());
		tasks_.pop();
	}
	if (task) {
		task();
	}
	return true;
}

void ThreadPool::WorkerLoop() {
	while (true) {
		std::function<void()> task;
		{
			std::unique_lock<std::mutex> lock(mutex_);
			cv_.wait(lock, [this] { return stop_.load(std::memory_order_relaxed) || !tasks_.empty(); });
			if (stop_.load(std::memory_order_relaxed) && tasks_.empty()) {
				return;
			}
			task = std::move(tasks_.front());
			tasks_.pop();
		}
		if (task) {
			task();
		}
	}
}

ThreadPool& GetThreadPool() {
	static ThreadPool pool;
	return pool;
}

} // namespace Common
