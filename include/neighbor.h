#pragma once

#include <cstddef>
#include <mutex>
#include <vector>
#include <limits>
#include "utils.h"

namespace pipeann {

  struct Neighbor {
    unsigned id;
    float distance;
    bool flag;
    bool visited;
    unsigned rev_id;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f), visited(false) {
    }
    Neighbor(unsigned id, float distance, bool f, unsigned r_id) : id{id}, distance{distance}, flag(f), visited(false), rev_id(r_id) {
    }

    inline bool operator<(const Neighbor &other) const {
      return (distance < other.distance) || (distance == other.distance && id < other.id);
    }
    inline bool operator==(const Neighbor &other) const {
      return (id == other.id);
    }
    inline bool operator>(const Neighbor &other) const {
      return (distance > other.distance) || (distance == other.distance && id > other.id);
    }
  };

  template<typename TagT = int>
  struct NeighborTag {
    TagT tag;
    float dist;
    NeighborTag() = default;

    NeighborTag(TagT tag, float dist) : tag{tag}, dist{dist} {
    }
    inline bool operator<(const NeighborTag &other) const {
      return (dist < other.dist) || (dist == other.dist && tag < other.tag);
    }
    inline bool operator==(const NeighborTag &other) const {
      return (tag == other.tag);
    }
  };

  typedef std::lock_guard<std::mutex> LockGuard;
  struct nhood {
    std::mutex lock;
    std::vector<Neighbor> pool;
    unsigned M;

    std::vector<unsigned> nn_old;
    std::vector<unsigned> nn_new;
    std::vector<unsigned> rnn_old;
    std::vector<unsigned> rnn_new;

    nhood() {
    }
    nhood(unsigned l, unsigned s, std::mt19937 &rng, unsigned N) {
      M = s;
      nn_new.resize(s * 2);
      GenRandom(rng, &nn_new[0], (unsigned) nn_new.size(), N);
      nn_new.reserve(s * 2);
      pool.reserve(l);
    }

    nhood(const nhood &other) {
      M = other.M;
      std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
      nn_new.reserve(other.nn_new.capacity());
      pool.reserve(other.pool.capacity());
    }
    void insert(unsigned id, float dist) {
      LockGuard guard(lock);
      if (dist > pool.front().distance)
        return;
      for (unsigned i = 0; i < pool.size(); i++) {
        if (id == pool[i].id)
          return;
      }
      if (pool.size() < pool.capacity()) {
        pool.push_back(Neighbor(id, dist, true));
        std::push_heap(pool.begin(), pool.end());
      } else {
        std::pop_heap(pool.begin(), pool.end());
        pool[pool.size() - 1] = Neighbor(id, dist, true);
        std::push_heap(pool.begin(), pool.end());
      }
    }

    template<typename C>
    void join(C callback) const {
      for (unsigned const i : nn_new) {
        for (unsigned const j : nn_new) {
          if (i < j) {
            callback(i, j);
          }
        }
        for (unsigned j : nn_old) {
          callback(i, j);
        }
      }
    }
  };

  struct SimpleNeighbor {
    unsigned id;
    float distance;

    SimpleNeighbor() = default;
    SimpleNeighbor(unsigned id, float distance) : id(id), distance(distance) {
    }

    inline bool operator<(const SimpleNeighbor &other) const {
      return (distance < other.distance) || (distance == other.distance && id < other.id);
    }

    inline bool operator==(const SimpleNeighbor &other) const {
      return id == other.id;
    }
  };
  struct SimpleNeighbors {
    std::vector<SimpleNeighbor> pool;
  };

  static inline unsigned InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
    // find the location to insert
    unsigned left = 0, right = K - 1;
    if (addr[left].distance > nn.distance) {
      memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
      addr[left] = nn;
      return left;
    }
    if (addr[right].distance < nn.distance) {
      addr[K] = nn;
      return K;
    }
    while (right > 1 && left < right - 1) {
      unsigned mid = (left + right) / 2;
      if (addr[mid].distance > nn.distance)
        right = mid;
      else
        left = mid;
    }
    // check equal ID

    while (left > 0) {
      if (addr[left].distance < nn.distance)
        break;
      if (addr[left].id == nn.id)
        return K + 1;
      left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id)
      return K + 1;
    memmove((char *) &addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
  }
  struct NeighborH {
    unsigned id;
    float distance1, distance2;
    bool flag;
    bool visited;
    int layer;

    NeighborH() = default;
    NeighborH(unsigned id, float distance1, float distance2, bool f, int layer) 
    : id{id}, distance1{distance1}, distance2{distance2}, flag(f), visited(false), layer(layer)
    { }

    inline bool operator<(const NeighborH &other) const {
      return (distance2 < other.distance2)
          || (distance2 == other.distance2 && distance1 < other.distance1)
          || (distance2 == other.distance2 && distance1 == other.distance1 && id < other.id);
    }
    inline bool operator==(const NeighborH &other) const {
      return (id == other.id);
    }
    // inline bool operator>(const NeighborH &other) const {
    //   return (distance > other.distance) || (distance == other.distance && id > other.id);
    // }
  };
  struct DEGNeighbor
  {
    unsigned id_;
    float emb_distance_;
    float geo_distance_;
    std::vector<std::pair<float, float>> available_range;
    unsigned layer_;

    DEGNeighbor() = default;
    DEGNeighbor(unsigned id, float emb_distance, float geo_distance) : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance)
    {
      available_range.emplace_back(0, 1);
    }
    DEGNeighbor(unsigned id, float emb_distance, float geo_distance, std::vector<std::pair<float, float>> range) : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance), available_range(range) {}
    DEGNeighbor(unsigned id, float emb_distance, float geo_distance, std::vector<std::pair<float, float>> range, unsigned l) : id_{id}, emb_distance_{emb_distance}, geo_distance_(geo_distance), available_range(range), layer_(l) {}

    inline bool operator<(const DEGNeighbor &other) const
    {
        // return geo_distance_ < other.geo_distance_;
        // 较小的 geo_distance_ 值会被排序到较前的位置
        return (geo_distance_ < other.geo_distance_ || (geo_distance_ == other.geo_distance_ && emb_distance_ < other.emb_distance_));
    }

    inline bool operator==(const DEGNeighbor &other) const {
      return (id_ == other.id_);
    }
    inline bool operator==(const unsigned &other) const {
      return (id_ == other);
    }
  };
  struct DEGSimpleNeighbor
  {
    unsigned id_;
    std::vector<std::pair<int8_t, int8_t>> active_range;

    DEGSimpleNeighbor() = default;
    DEGSimpleNeighbor(unsigned id, std::vector<std::pair<int8_t, int8_t>> range) : id_{id}, active_range(range) {}
  };
}  // namespace pipeann
