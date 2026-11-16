#pragma once
#include <atomic>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#ifndef HVS_REVERSE_ALWAYS_PRUNE
#define HVS_REVERSE_ALWAYS_PRUNE 0
#endif
namespace hvs_ab {
inline std::atomic<uint64_t> insert_calls{0}, reverse_targets{0}, reprune_calls{0}, cleanup_changes{0};
}
extern "C" void hvs_ab_write_stats(const char *path) {
 std::ofstream f(path);
 f<<"{\"reverse_always_prune\":"<<HVS_REVERSE_ALWAYS_PRUNE<<",\"insert_calls\":"<<hvs_ab::insert_calls.load()<<",\"reverse_targets\":"<<hvs_ab::reverse_targets.load()<<",\"B_reprunes\":"<<hvs_ab::reprune_calls.load()<<",\"final_cleanup_changed_nodes\":"<<hvs_ab::cleanup_changes.load()<<"}\n";
 if(!f)throw std::runtime_error("AB stats write failed");
}
