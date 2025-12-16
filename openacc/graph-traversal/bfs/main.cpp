#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <climits>
#include <cctype>

namespace fs = std::filesystem;


double bfs_serial(const int* row_ptr, const int* col_idx, int n, int src, int* cost);
double bfs_openacc(const int* row_ptr, const int* col_idx, int n, int src, int* cost);


struct CSR {
  int n = 0;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
};

static inline std::string trim(const std::string& s){
  size_t a=0,b=s.size();
  while(a<b && std::isspace((unsigned char)s[a])) ++a;
  while(b>a && std::isspace((unsigned char)s[b-1])) --b;
  return s.substr(a,b-a);
}

static bool load_mtx_to_csr(const std::string& path, CSR& out){
  std::ifstream fin(path);
  if(!fin) return false;

  std::string line;
  std::getline(fin,line);
  bool symmetric = (line.find("symmetric")!=std::string::npos);

  long long M=0,N=0,NZ=0;
  while(std::getline(fin,line)){
    line=trim(line);
    if(line.empty()||line[0]=='%') continue;
    std::istringstream iss(line);
    iss>>M>>N>>NZ;
    break;
  }

  std::vector<std::pair<int,int>> edges;
  edges.reserve(NZ*(symmetric?2:1));

  long long read=0;
  while(read<NZ && std::getline(fin,line)){
    line=trim(line);
    if(line.empty()||line[0]=='%') continue;
    long long i,j;
    std::istringstream iss(line);
    iss>>i>>j;
    edges.emplace_back(i-1,j-1);
    if(symmetric && i!=j) edges.emplace_back(j-1,i-1);
    read++;
  }

  out.n = std::max((long long)M,(long long)N);
  std::vector<int> deg(out.n,0);
  for(auto&e:edges) deg[e.first]++;

  out.row_ptr.assign(out.n+1,0);
  for(int i=0;i<out.n;i++) out.row_ptr[i+1]=out.row_ptr[i]+deg[i];
  out.col_idx.assign(out.row_ptr.back(),0);

  std::vector<int> cur=out.row_ptr;
  for(auto&e:edges) out.col_idx[cur[e.first]++]=e.second;

  return true;
}

static std::vector<fs::path> find_graphs(){
  std::vector<fs::path> v;
  fs::path data="../data";
  if(!fs::exists(data)) return v;
  for(auto&d:fs::directory_iterator(data)){
    fs::path m=d.path()/(d.path().filename().string()+".mtx");
    if(fs::exists(m)) v.push_back(m);
  }
  std::sort(v.begin(),v.end());
  return v;
}

static void append_csv(const fs::path& csv,const std::string& name,
                       int n,long long m,const std::vector<double>& ms){
  bool exists=fs::exists(csv);
  std::ofstream f(csv,std::ios::app);
  if(!exists){
    f<<"graph,num_nodes,num_edges,avg_degree,"
     <<"ms_t1,ms_t2,ms_t4,ms_t8,ms_t16,ms_t32,"
     <<"mteps_t1,mteps_t2,mteps_t4,mteps_t8,mteps_t16,mteps_t32,"
     <<"speedup_t2,speedup_t4,speedup_t8,speedup_t16,speedup_t32\n";
  }
  double t1=ms[0];
  auto mteps=[&](double t){return t>0?(m/(t/1000.0))/1e6:0.0;};
  auto sp=[&](double t){return t1>0?(t1/t):0.0;};

  f<<name<<","<<n<<","<<m<<","<<(double)m/n<<","
   <<ms[0]<<","<<ms[1]<<","<<ms[2]<<","<<ms[3]<<","<<ms[4]<<","<<ms[5]<<","
   <<mteps(ms[0])<<","<<mteps(ms[1])<<","<<mteps(ms[2])<<","<<mteps(ms[3])<<","<<mteps(ms[4])<<","<<mteps(ms[5])<<","
   <<sp(ms[1])<<","<<sp(ms[2])<<","<<sp(ms[3])<<","<<sp(ms[4])<<","<<sp(ms[5])<<"\n";
}

int main(int argc,char**argv){
  bool do_c=false,do_p=false;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-c")) do_c=true;
    if(!strcmp(argv[i],"-p")) do_p=true;
  }

  fs::create_directories("../output");

  if(do_c){
    CSR G;
    load_mtx_to_csr("../data/roadNet-CA/roadNet-CA.mtx",G);

    std::vector<int> ref(G.n),par(G.n);
    double tser=bfs_serial(G.row_ptr.data(),G.col_idx.data(),G.n,0,ref.data());
    double tgpu=bfs_openacc(G.row_ptr.data(),G.col_idx.data(),G.n,0,par.data());

    bool ok=true;
    for(int i=0;i<G.n;i++) if(ref[i]!=par[i]){ok=false;break;}

    printf("serial %.3f ms\n",tser);
    printf("openacc %.3f ms %s\n",tgpu,ok?"OK":"MISMATCH");
  }

  if(do_p){
    auto files=find_graphs();
    fs::path csv="../output/bfs_profile.csv";

    for(auto&mtx:files){
      CSR G;
      load_mtx_to_csr(mtx.string(),G);
      std::vector<double> ms(6);

      for(int i=0;i<6;i++){
        std::vector<int> cost(G.n);
        ms[i]=bfs_openacc(G.row_ptr.data(),G.col_idx.data(),G.n,0,cost.data());
      }

      append_csv(csv,mtx.parent_path().filename().string(),
                 G.n,(long long)G.col_idx.size(),ms);

      printf("[-p] %s done\n",mtx.parent_path().filename().c_str());
    }
    printf("CSV -> %s\n",csv.c_str());
  }
  return 0;
}
