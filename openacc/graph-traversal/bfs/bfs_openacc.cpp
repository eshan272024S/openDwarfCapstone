#include <vector>
#include <chrono>

double bfs_openacc(const int* row_ptr, const int* col_idx, int n, int src, int* cost){
  if(n<=0 || !row_ptr || !col_idx || !cost) return 0.0;
  if(src<0 || src>=n) return 0.0;

  std::vector<int> graph_mask_v(n,0);
  std::vector<int> updating_mask_v(n,0);
  std::vector<int> visited_v(n,0);

  for(int i=0;i<n;i++) cost[i]=-1;

  graph_mask_v[src]=1;
  visited_v[src]=1;
  cost[src]=0;

  int* graph_mask = graph_mask_v.data();
  int* updating_mask = updating_mask_v.data();
  int* visited = visited_v.data();

  using clk=std::chrono::high_resolution_clock;
  auto t0=clk::now();

  int over;

#pragma acc data copyin(row_ptr[0:n+1], col_idx[0:row_ptr[n]]) \
                 copy(cost[0:n]) \
                 copy(graph_mask[0:n], updating_mask[0:n], visited[0:n])
  {
    do{
      over=0;

#pragma acc parallel loop
      for(int tid=0; tid<n; tid++){
        if(graph_mask[tid]){
          graph_mask[tid]=0;
          int c = cost[tid];
          for(int i=row_ptr[tid]; i<row_ptr[tid+1]; i++){
            int id = col_idx[i];
            if(visited[id]==0){
              cost[id]=c+1;
              updating_mask[id]=1;
            }
          }
        }
      }

#pragma acc parallel loop reduction(||:over)
      for(int tid=0; tid<n; tid++){
        if(updating_mask[tid]){
          graph_mask[tid]=1;
          visited[tid]=1;
          updating_mask[tid]=0;
          over=1;
        }
      }

#pragma acc wait
    } while(over);
  }

  auto t1=clk::now();
  return std::chrono::duration<double,std::milli>(t1-t0).count();
}
