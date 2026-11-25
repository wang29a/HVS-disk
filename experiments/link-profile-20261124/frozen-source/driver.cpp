#include "index.h"
#include "parameters.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <shared_mutex>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

namespace {
using Id = uint32_t;
using Edge = pipeann::DEGNeighbor;
using Graph = std::vector<std::vector<Edge>>;
using Interval = std::pair<float,float>;
using Clock = std::chrono::steady_clock;
constexpr size_t page_bytes = 8192, dim = 512;
constexpr unsigned max_degree = 40, records_per_page = 25, row_bytes = 324;

void check(bool ok, const char *msg) {
  if (!ok)
    throw std::runtime_error(msg);
}

double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

template<class T> T get(std::istream &f) {
  T value;
  check(bool(f.read(reinterpret_cast<char *>(&value),sizeof(T))),"truncated float graph");
  return value;
}

template<class T> void put(std::ostream &f, const T &value) {
  f.write(reinterpret_cast<const char *>(&value),sizeof(value));
}

void graph_header(std::ostream &f, size_t n) {
  put(f,uint64_t(n));
  put(f,uint32_t(dim));
  put(f,uint32_t(dim));
}

void graph_header(std::istream &f, size_t n) {
  check(get<uint64_t>(f)==n && get<uint32_t>(f)==dim && get<uint32_t>(f)==dim,
        "unexpected float graph dimensions");
}

void put_row(std::ostream &f, const std::vector<Edge> &row) {
  put(f,uint32_t(row.size()));
  for (const auto &e : row) {
    put(f,e.id_);
    put(f,uint32_t(e.available_range.size()));
    for (auto p : e.available_range) {
      put(f,p.first);
      put(f,p.second);
    }
  }
}

std::vector<Edge> read_row(std::istream &f, const std::vector<Id> &ids, Id self) {
  const Id degree = get<Id>(f);
  check(degree<=max_degree,"shard degree");
  std::vector<Edge> row;
  row.reserve(degree);
  for (Id i=0; i<degree; ++i) {
    const Id local = get<Id>(f);
    const Id count = get<Id>(f);
    check(local<ids.size() && ids[local]!=self && count>0 && count<=1024,"invalid shard edge");
    std::vector<Interval> intervals;
    intervals.reserve(count);
    for (Id j=0; j<count; ++j) {
      const float a=get<float>(f), b=get<float>(f);
      check(std::isfinite(a) && std::isfinite(b) && 0<=a && a<=b && b<=1,"invalid float interval");
      intervals.emplace_back(a,b);
    }
    row.emplace_back(ids[local],0.f,0.f,std::move(intervals),0);
  }
  return row;
}

std::vector<Id> read_ids(const std::string &name, Id n) {
  std::ifstream f(name,std::ios::binary);
  std::vector<Id> result;
  Id id;
  while (f.read(reinterpret_cast<char *>(&id),sizeof(id))) {
    check(id<n && (result.empty() || result.back()<id),"membership must be sorted and unique");
    result.push_back(id);
  }
  check(f.eof() && !f.gcount() && !result.empty(),"membership list");
  return result;
}

struct Shard {
  std::vector<Id> ids;
  std::ifstream graph;
  size_t at=0;
  Shard(const std::string &membership,const std::string &path,Id n)
      : ids(read_ids(membership,n)),graph(path,std::ios::binary) {
    graph_header(graph,ids.size());
  }
  bool contains(Id id) const { return at<ids.size() && ids[at]==id; }
  std::vector<Edge> get_row(Id id) {
    ++at;
    return read_row(graph,ids,id);
  }
  void end() { check(at==ids.size() && graph.peek()==EOF,"incomplete shard consumed"); }
};

std::vector<std::unique_ptr<Shard>> shard_inputs(const char *path,Id n) {
  std::vector<std::unique_ptr<Shard>> shards;
  std::ifstream plan(path);
  std::string ids, graph;
  while (plan>>ids>>graph)
    shards.emplace_back(new Shard(ids,graph,n));
  check(plan.eof() && !shards.empty(),"no shards in merge plan");
  return shards;
}

#ifdef HVS_LINK_PROFILE
unsigned profile_L = 200, profile_C = 750;
#endif
void set_params(pipeann::Parameters &p, unsigned r) {
  p.Set<unsigned>("R",r);
#ifdef HVS_LINK_PROFILE
  p.Set<unsigned>("L",profile_L);
  p.Set<unsigned>("C",profile_C);
#else
  p.Set<unsigned>("L",200);
  p.Set<unsigned>("C",750);
#endif
  p.Set<float>("alpha",1.2f);
  p.Set<bool>("saturate_graph",r==max_degree);
  p.Set<unsigned>("num_threads",8);
}

void build_shard(const char *x,const char *y,Id n,unsigned r,const std::string &out) {
  pipeann::Parameters p;
  set_params(p,r);
  p.Set<std::string>("save_path",out+".builder");
  pipeann::HybridIndex<float> index(pipeann::Metric::L2,dim,dim,n,false,false);
  index.build(x,y,n,p);
  check(index._final_graph.size()>=n,"builder did not return all rows");
  std::ofstream f(out,std::ios::binary);
  graph_header(f,n);
  for (Id id=0;id<n;++id) {
    check(index._final_graph[id].size()<=r,"invalid shard degree");
    put_row(f,index._final_graph[id]);
  }
  f.close();
  check(bool(f),"shard graph output");
}

void relabel(Id n,const char *manifest,const char *out) {
  auto shards=shard_inputs(manifest,n);
  std::ofstream f(out,std::ios::binary);
  graph_header(f,n);
  for (Id id=0;id<n;++id) {
    unsigned found=0;
    for (auto &s:shards)
      if (s->contains(id)) {
        put_row(f,s->get_row(id));
        ++found;
      }
    check(found==1,"legacy partition requires exactly one membership per space");
  }
  for (auto &s:shards)
    s->end();
  f.close();
  check(bool(f),"legacy per-space relabel output");
}

void read_coordinates(const char *path,Id n,float *data) {
  std::ifstream f(path,std::ios::binary);
  check(get<Id>(f)==n && get<Id>(f)==dim,"coordinate matrix dimensions");
  check(bool(f.read(reinterpret_cast<char *>(data),size_t(n)*dim*sizeof(float))) && f.peek()==EOF,
        "coordinate matrix payload");
}

struct Geometry {
  pipeann::HybridIndex<float> &index;
  std::pair<float,float> distance(Id a,Id b) const {
    float x=index._distance->compare(index._data1+size_t(a)*dim,index._data1+size_t(b)*dim,dim);
    float y=index._distance->compare(index._data2+size_t(a)*dim,index._data2+size_t(b)*dim,dim);
    return {std::sqrt(std::max(0.f,x)),std::sqrt(std::max(0.f,y))};
  }
  std::vector<Id> entries(Id n) const {
    std::array<double,dim> ax{},ay{};
    for (Id i=0;i<n;++i)
      for (size_t j=0;j<dim;++j) {
        ax[j]+=index._data1[dim*size_t(i)+j];
        ay[j]+=index._data2[dim*size_t(i)+j];
      }
    std::array<float,dim> x{},y{};
    for (size_t j=0;j<dim;++j) {
      x[j]=ax[j]/n;
      y[j]=ay[j]/n;
    }
    std::vector<pipeann::NeighborH> skyline;
    std::vector<Id> ids;
    std::shared_mutex mutex;
    for (Id i=0;i<n;++i)
      index.UpdateEnterpointSet(skyline,ids,x.data(),y.data(),i,mutex);
    if (ids.empty())
      ids.push_back(0);
    return ids;
  }
};

using Quantized = std::pair<uint8_t,uint8_t>;
std::vector<Quantized> merge2(const std::vector<Interval> &ranges) {
  std::vector<Quantized> parts;
  for (auto p:ranges) {
    unsigned lo=unsigned(std::floor(100.*p.first)),hi=unsigned(std::ceil(100.*p.second));
    if (!hi)
      hi=1;
    parts.emplace_back(lo,hi);
  }
  std::sort(parts.begin(),parts.end());
  std::vector<Quantized> merged;
  for (auto p:parts) {
    if (!merged.empty() && p.first<=merged.back().second)
      merged.back().second=std::max(merged.back().second,p.second);
    else
      merged.push_back(p);
  }
  while (merged.size()>2) {
    size_t best=0;
    for (size_t i=1;i+1<merged.size();++i)
      if (int(merged[i+1].first)-merged[i].second<int(merged[best+1].first)-merged[best].second)
        best=i;
    merged[best].second=merged[best+1].second;
    merged.erase(merged.begin()+best+1);
  }
  for (auto p:ranges) {
    bool covered=false;
    for (auto v:merged)
      covered |= v.first/100.<=p.first && v.second/100.>=p.second;
    check(covered,"Merge-2 narrowed an edge interval");
  }
  check(!merged.empty(),"edge without an active interval");
  return merged;
}

void graph_stats(const Graph &g, const std::vector<Id> &entries,std::ostream &out) {
  std::array<uint64_t,max_degree+1> degrees{};
  std::vector<uint64_t> interval_counts(1,0);
  uint64_t edges=0;
  for (Id u=0;u<g.size();++u) {
    const auto &row=g[u];
    check(row.size()<=max_degree,"merged degree cap");
    ++degrees[row.size()];
    edges+=row.size();
    std::vector<Id> seen;
    for (const auto &e:row) {
      check(e.id_<g.size() && e.id_!=u && std::find(seen.begin(),seen.end(),e.id_)==seen.end(),
            "invalid or duplicate output edge");
      seen.push_back(e.id_);
      const auto p=e.available_range.size();
      if (p>=interval_counts.size())
        interval_counts.resize(p+1);
      ++interval_counts[p];
    }
  }
  std::vector<Id> reached;
  std::vector<uint8_t> mark(g.size());
  for (Id u:entries) {
    check(u<g.size(),"entry ID");
    if (!mark[u]) {
      reached.push_back(u);
      mark[u]=1;
    }
  }
  for (size_t i=0;i<reached.size();++i)
    for (auto &e:g[reached[i]])
      if (!mark[e.id_]) {
        mark[e.id_]=1;
        reached.push_back(e.id_);
      }
  auto hist=[&](const auto &values) {
    out<<"[";
    for (size_t i=0;i<values.size();++i)
      out<<(i?",":"")<<values[i];
    out<<"]";
  };
  out<<std::setprecision(12)<<"{\"nodes\":"<<g.size()<<",\"edges\":"<<edges
     <<",\"mean_degree\":"<<double(edges)/g.size()<<",\"unreachable\":"<<g.size()-reached.size()
     <<",\"reachable_fraction\":"<<double(reached.size())/g.size()<<",\"entry_count\":"<<entries.size()
     <<",\"degree_histogram\":";
  hist(degrees);
  out<<",\"float_interval_histogram\":";
  hist(interval_counts);
  out<<",\"validated\":true}\n";
}

void save_merge(const Graph &g,const std::vector<Id> &entries,const std::string &root) {
  std::ofstream audit(root+"/graph-audit.json");
  graph_stats(g,entries,audit);
  audit.close();
  check(bool(audit),"graph audit write");
  std::ofstream raw(root+"/float-graph.bin",std::ios::binary);
  graph_header(raw,g.size());
  for (auto &row:g)
    put_row(raw,row);
  raw.close();
  check(bool(raw),"merged float graph write");

  std::array<uint8_t,page_bytes> meta{};
  const Id n=g.size(), d=dim, r=max_degree, m=2, ne=entries.size();
  const uint64_t per_page=records_per_page;
  check(ne>0 && 32+4*ne<=meta.size(),"entry metadata capacity");
  std::memcpy(meta.data(),&n,4);
  std::memcpy(meta.data()+4,&d,4);
  std::memcpy(meta.data()+8,&d,4);
  std::memcpy(meta.data()+12,&r,4);
  std::memcpy(meta.data()+16,&m,4);
  std::memcpy(meta.data()+20,&per_page,8);
  std::memcpy(meta.data()+28,&ne,4);
  std::memcpy(meta.data()+32,entries.data(),4*ne);
  std::ofstream metadata(root+"/_disk.index",std::ios::binary),topology(root+"/disk_index_graph",std::ios::binary);
  metadata.write(reinterpret_cast<char *>(meta.data()),meta.size());
  for (size_t start=0;start<g.size();start+=records_per_page) {
    std::array<uint8_t,page_bytes> page{};
    for (size_t j=0;j<records_per_page && start+j<g.size();++j) {
      const auto &row=g[start+j];
      auto *record=page.data()+j*row_bytes;
      const Id d0=row.size();
      std::memcpy(record,&d0,4);
      for (size_t k=0;k<row.size();++k) {
        auto *p=record+4+k*8;
        std::memcpy(p,&row[k].id_,4);
        auto ranges=merge2(row[k].available_range);
        for (size_t z=0;z<ranges.size();++z) {
          p[4+2*z]=ranges[z].first;
          p[5+2*z]=ranges[z].second;
        }
      }
    }
    topology.write(reinterpret_cast<char *>(page.data()),page.size());
  }
  metadata.close();
  topology.close();
  check(bool(metadata) && bool(topology),"Merge-2 write");
}

void audit_existing(Id n,const char *input_graph,const char *metadata,const char *output) {
  std::ifstream f(input_graph,std::ios::binary),m(metadata,std::ios::binary);
  graph_header(f,n);
  m.seekg(28);
  const Id count=get<Id>(m);
  check(count && count<2041,"reference entry count");
  std::vector<Id> entries(count),ids(n);
  for (auto &e:entries)
    e=get<Id>(m);
  std::iota(ids.begin(),ids.end(),0);
  Graph g(n);
  for (Id u=0;u<n;++u)
    g[u]=read_row(f,ids,u);
  check(f.peek()==EOF,"trailing reference graph");
  std::ofstream out(output);
  graph_stats(g,entries,out);
}

void merge(const char *x,const char *y,Id n,const char *manifest,const std::string &out) {
  auto start=Clock::now();
  auto shards=shard_inputs(manifest,n);
  pipeann::HybridIndex<float> index(pipeann::Metric::L2,dim,dim,n,false,false);
  check(index._aligned_dim1==dim && index._aligned_dim2==dim,"aligned coordinate dimensions");
  index._final_graph.resize(n);
  read_coordinates(x,n,index._data1);
  read_coordinates(y,n,index._data2);
  Geometry geo{index};
  pipeann::Parameters p;
  set_params(p,max_degree);
  auto scan=Clock::now();
  std::vector<uint8_t> multi(n);
  uint64_t duplicates=0;
  constexpr size_t tile=4096;
  for (Id first=0;first<n;first+=tile) {
    const Id end=std::min<Id>(n,first+tile);
    for (Id u=first;u<end;++u) {
      unsigned copies=0;
      auto &row=index._final_graph[u];
      for (auto &s:shards) {
        if (!s->contains(u))
          continue;
        auto other=s->get_row(u);
        if (copies++==0)
          row=std::move(other);
        else
          row.insert(row.end(),std::make_move_iterator(other.begin()),std::make_move_iterator(other.end()));
      }
      check(copies,"missing input node during merge");
      multi[u]=copies>1;
    }
#pragma omp parallel for schedule(dynamic,32) reduction(+ : duplicates)
    for (Id u=first;u<end;++u) {
      auto &row=index._final_graph[u];
      if (!multi[u]) {
        for (auto &e:row) {
          auto d=geo.distance(u,e.id_);
          e.emb_distance_=d.first;
          e.geo_distance_=d.second;
        }
        continue;
      }
      std::vector<Id> ids;
      ids.reserve(row.size());
      for (auto &e:row)
        ids.push_back(e.id_);
      std::sort(ids.begin(),ids.end());
      ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
      duplicates+=row.size()-ids.size();
      std::vector<pipeann::NeighborH> pool;
      pool.reserve(ids.size());
      for (Id v:ids) {
        auto d=geo.distance(u,v);
        pool.emplace_back(v,d.first,d.second,true,0);
      }
      std::vector<Edge> chosen;
      index.prune_neighbors_no_opt(pool,p,chosen);
      row.swap(chosen);
    }
    if (first % (tile*32)==0)
      std::cerr<<"MERGE_ROWS "<<end<<"/"<<n<<"\n";
  }
  for (auto &s:shards)
    s->end();
  double reselect_seconds=seconds(scan);
  auto reversing=Clock::now();
  size_t multi_nodes=0;
  for (Id u=0;u<n;++u)
    if (multi[u]) {
      ++multi_nodes;
      auto outgoing=index._final_graph[u];
      index.inter_insert(u,outgoing,p);
    }
  double reverse_seconds=seconds(reversing);
  auto export_start=Clock::now();
  auto entries=geo.entries(n);
  save_merge(index._final_graph,entries,out);
  std::ofstream timers(out+"/merge-times.json");
  timers<<std::setprecision(12)<<"{\"merged_nodes\":"<<multi_nodes<<",\"merged_fraction\":"<<double(multi_nodes)/n
        <<",\"deduplicated_candidate_edges\":"<<duplicates
        <<",\"row_load_and_reselection_seconds\":"<<reselect_seconds
        <<",\"reverse_edges_seconds\":"<<reverse_seconds
        <<",\"entry_and_export_seconds\":"<<seconds(export_start)<<",\"wall_seconds\":"<<seconds(start)<<"}\n";
}
} // namespace

int main(int argc,char **argv) {
  try {
    omp_set_dynamic(0);
    omp_set_num_threads(8);
    const std::string mode=argc>1?argv[1]:"";
#ifdef HVS_LINK_PROFILE
    if (argc==9 && mode=="shard") {
      profile_L = std::stoul(argv[7]);
      profile_C = std::stoul(argv[8]);
      argc = 7;
    }
#endif
    if (argc==7 && mode=="shard")
      build_shard(argv[2],argv[3],std::stoul(argv[4]),std::stoul(argv[5]),argv[6]);
    else if (argc==5 && mode=="relabel")
      relabel(std::stoul(argv[2]),argv[3],argv[4]);
    else if (argc==7 && mode=="merge")
      merge(argv[2],argv[3],std::stoul(argv[4]),argv[5],argv[6]);
    else if (argc==6 && mode=="audit")
      audit_existing(std::stoul(argv[2]),argv[3],argv[4],argv[5]);
    else
      throw std::runtime_error("shard/relabel/merge/audit: argument count");
    return 0;
  } catch (const std::exception &e) {
    std::cerr<<e.what()<<std::endl;
    return 1;
  }
}
