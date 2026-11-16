// Input/output and parameters only; author Link/search/prune stay in original TUs.
#include "component.h"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
template<class T> void put(std::ostream &o,const T &v){o.write((const char*)&v,sizeof(v));}
std::vector<float> matrix(const char *path,unsigned &n,unsigned &d){
 std::ifstream f(path,std::ios::binary);f.read((char*)&n,4);f.read((char*)&d,4);std::vector<float> a(size_t(n)*d);if(!f.read((char*)a.data(),a.size()*4) || f.peek()!=EOF)throw std::runtime_error("matrix");return a;
}
int main(int argc,char **argv){
 try {
  if(argc!=4)throw std::runtime_error("base1 base2 output-float-graph");
  unsigned n,d,n2,d2;auto x=matrix(argv[1],n,d),y=matrix(argv[2],n2,d2);if(n!=n2)throw std::runtime_error("paired bases");
  auto *index=new stkq::Index(1.f,1.f);
  index->setBaseEmbData(x.data());index->setBaseLocData(y.data());index->setBaseLen(n);index->setBaseEmbDim(d);index->setBaseLocDim(d2);
  stkq::Parameters p;p.set<unsigned>("max_m",40);p.set<unsigned>("max_m0",40);p.set<unsigned>("ef_construction",200);p.set<unsigned>("n_threads",1);p.set<int>("mult",0);p.set<unsigned>("C",750);index->setParam(p);
  omp_set_num_threads(1);auto *init=new stkq::ComponentInitDEG(index);init->InitInner();
  std::ofstream out(argv[3],std::ios::binary);uint64_t nn=n;put(out,nn);put(out,d);put(out,d2);
  for(unsigned u=0;u<n;++u){if(!index->DEG_nodes_[u])throw std::runtime_error("missing author node");auto &edges=index->DEG_nodes_[u]->GetFriends();unsigned degree=edges.size();put(out,degree);for(const auto &e:edges){put(out,e.id_);unsigned nr=e.available_range.size();put(out,nr);for(auto r:e.available_range){put(out,r.first);put(out,r.second);}}}
  out.close();if(!out)throw std::runtime_error("author graph write");std::cout<<"AUTHOR_GRAPH_COMPLETE nodes="<<n<<"\n";std::exit(0);
 }catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}
}
