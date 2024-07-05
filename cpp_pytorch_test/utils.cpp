#include <torch/extension.h>
#include <torch/torch.h>
#include <random>
#include <vector>
#include <queue>
#include <utility>
#include <functional>

using namespace std;

at::Tensor getFirstElement(at::Tensor tensor) {
  //at::Tensor& firstElement = tensor[0];
  //firstElement = 90;
  return tensor.index({0,1});

}

class candidate {
  public:
    float score;
    int64_t index;
    int64_t left_selected;
    int64_t right_selected;
    candidate(float score=0, int64_t index=0, int64_t left_selected=0, int64_t right_selected=0)
    {
      this->score = score;
      this->index = index;
      this->left_selected = left_selected;
      this->right_selected = right_selected;
    }
    
    //_candidate(float score_value, int64_t index_value) : score(score_value), index(index_value) // Create an object of type _book.
};

struct binary_search_result
{
  bool suc;
  int index;
};


bool operator <(const candidate& lhs, const candidate& rhs) {
    return lhs.score < rhs.score; // descending order by value
}

struct sorted_point_cloud
{
    at::Tensor projected_values;
    at::Tensor order;
};




//  float firstElement = tensor[0].item().toFloat();


sorted_point_cloud project_and_sort(at::Tensor xyz)
{
    torch::Tensor xyz_sum = torch::sum(xyz, 2);
    at::Tensor sorted_xyz, order;
    std::tie(sorted_xyz, order) = xyz_sum.sort();
    sorted_point_cloud result;
    result.projected_values = sorted_xyz;
    result.order = order;
    return result;
}
/*def project_and_sort(xyz):
  num_points = xyz.shape[1]
  projected_values = np.sum(xyz, 2)
  projected_values_ = np.sort(projected_values)
  order = np.argsort(projected_values)
  #projected_values, order = projected_values.sort()
  return (projected_values_, order)*/

int test_working_with_queue()
{
  priority_queue<candidate, vector<candidate>, less<candidate>> pq;
  int result;
  
  //candidate temp = {1,5};
  //pq.push(temp);
  //candidate A(1,5);
  //pq.push(candidate(1,5));
  pq.push({10, 5, 90, -34});
  pq.push({-2, 3, 0, 0});
  pq.push({35, 7, 9012, -7834});
  pq.push({0.4, 2, -12, -12});
  
      while (!pq.empty()) {
        auto pair = pq.top();
        cout << "Key: " << pair.score << ", Value: " << pair.left_selected << endl;
        result = pair.score;
        pq.pop();
    } 
    return result;
}

/*
def binary_search(projected, left, right, query):
  middle = int((left+right)/2)
  if right < left:
    return (0, left)
  if query == projected[0,middle]:
    return (1, middle)
  elif query< projected[0,middle]:
    return binary_search(projected, left, middle-1, query)
  elif query> projected[0,middle]:
    return binary_search(projected, middle+1, right, query)

*/

binary_search_result binary_search(at::Tensor& projected, int64_t left, int64_t right, float query)
{
  int middle = (int)((left+right)/2);
  binary_search_result result;
  if (right < left)
  {
    result.suc = 0;
    result.index = left;
    return result;
  }
  if (query == projected[0][middle].item().toFloat())
  {
    result.suc = 1;
    result.index = middle;
    return result;
  }
  else if (query< projected[0][middle].item().toFloat())
  {
    return binary_search(projected, left, middle-1, query);
  }
  else if (query> projected[0][middle].item().toFloat())
  {
    return binary_search(projected, middle+1, right, query);
  }
}

candidate find_middle_candidate(at::Tensor&projected, int64_t left, int64_t right)
{
  float query = ((projected[0][left] + projected[0][right])/2).item().toFloat();
  binary_search_result temp = binary_search(projected, left, right, query);
  bool suc = temp.suc;
  int64_t res = temp.index;
  candidate result;
  if (suc)
  {
    result.index = res;
    result.score = (abs(projected[0][res] - projected[0][left])).item().toFloat();
    return result;
  }
  else if (res == right +1)
  {
    result.index = right;
    result.score = 0;
    return result;
  }
  else if (res == 0)
  {
    result.index = 0;
    result.score = 0;
    return result;
  }
  else
  {
    if ((abs(projected[0][res-1] - query)).item().toFloat() <= (abs(projected[0][res]- projected[0][right])).item().toFloat())
    {
          result.index = res - 1;
          result.score = (abs(projected[0][res-1] - projected[0][left])).item().toFloat();
          return result;
    }
    else
    {
          result.index = res;
          result.score = (abs(projected[0][res] - projected[0][right])).item().toFloat();
          return result;
    }
  }
}

float test_for_find_middle_candidate_score(at::Tensor&projected, int64_t left, int64_t right)
{
  candidate result = find_middle_candidate(projected, left, right);
  return result.score;
}

int test_for_find_middle_candidate_index(at::Tensor&projected, int64_t left, int64_t right)
{
  candidate result = find_middle_candidate(projected, left, right);
  return result.index;
}

bool test_for_binary_search_score(at::Tensor& projected, int64_t left, int64_t right, float query)
{
  binary_search_result result = binary_search(projected, left, right, query);
  return result.suc;
}

int64_t test_for_binary_search_index(at::Tensor& projected, int64_t left, int64_t right, float query)
{
  binary_search_result result = binary_search(projected, left, right, query);
  return result.index;
}


torch::Tensor farthest_point_sampling(torch::Tensor& xyz, int64_t npoint)
{
  
  sorted_point_cloud sorted = project_and_sort(xyz);
  
  int centroids_count = 0;
  at::Tensor projected_values = sorted.projected_values;
  at::Tensor order = sorted.order;
  
  int64_t B = xyz.size(0);
  int64_t N = xyz.size(1);
  
  torch::Tensor centroids = torch::zeros({npoint}, torch::dtype(torch::kLong));
  auto out_data = centroids.data_ptr<int64_t>();
  
  //vector <int64_t> selected_points;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int64_t> dis(2, N-1);
  

  //centroids.at({centroids_count++}) = dis(gen);
  int64_t first_selected = dis(gen);
  out_data[centroids_count++] = first_selected;
  int64_t centroids_0 = first_selected;
  //int64_t centroids_0 = centroids[0].item().toInt();
  cout<<endl<<"function is called"<<endl;
  float head_canidate_score = abs(projected_values[0][centroids_0] - projected_values[0][0]).item().toFloat();
  float tail_candidate_score = abs(projected_values[0][centroids_0] - projected_values[0][N-1]).item().toFloat();
  
  priority_queue<candidate, vector<candidate>, less<candidate>> candidates;
  //candidates.push({10                     , 5, 90, -34               });
  candidate temp1(-1 *head_canidate_score, 0, -2, centroids_0);
  candidate temp2(-1 *tail_candidate_score, N-1, centroids_0, -1);
  candidates.push(temp1);
  candidates.push(temp2);
  
  for (int i = 0; i < npoint -1; i++)
  {
     candidate temp_candidate = candidates.top();
     int64_t next_selected = temp_candidate.index;
     int64_t left_selected = temp_candidate.left_selected;
     int64_t right_selected = temp_candidate.right_selected;

     //centroids.at({centroids_count++}) = next_selected;
     out_data[centroids_count++] = next_selected;
     //selected_points.push_back(next_selected);
     // Adding the right candidate:
     if (!(right_selected == -1 || right_selected==next_selected+1))
     {
      candidate temp = find_middle_candidate(projected_values, next_selected, right_selected);
      int64_t middle = temp.index;
      float score = temp.score;
      candidate new_candidate;
      /*new_candidate.score = -1 * score;
      new_candidate.index = middle;
      new_candidate.left_selected = next_selected;
      new_candidate.right_selected = right_selected;*/
      candidates.push({-1*score, middle, next_selected, right_selected});
     }
     // Adding the left candidate:
     if (!(left_selected == -2 || left_selected == next_selected -1))
     {
      candidate temp = find_middle_candidate(projected_values, left_selected, next_selected);
      int64_t middle = temp.index;
      float score = temp.score;
      candidates.push({-1 * score, middle, left_selected, next_selected});
     }
     
     //centroids.index_put_(torch:: tensor({0, torch::arange(selected_points.size(1)).to(torch::long)}, order.index_select(1, selected_points));
  }
  return centroids;
  //return head_canidate_score - tail_candidate_score;

}



at::Tensor& modifyTensor(at::Tensor& tensor) {
  tensor[0] = 90;
  return tensor;
}


torch::Tensor trilinear_fw_test(
    const torch::Tensor feats,
    const torch::Tensor points
)
{
    return feats;

}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_fw_test", &trilinear_fw_test);
    m.def("getFirstElement", &getFirstElement);
    m.def("modifyTensor", &modifyTensor);
    m.def("project_and_sort", &project_and_sort);
    m.def("farthest_point_sampling", &farthest_point_sampling);
    m.def("test_working_with_queue", &test_working_with_queue);
    m.def("test_for_binary_search_score", &test_for_binary_search_score);
    m.def("test_for_binary_search_index", &test_for_binary_search_index);
    m.def("test_for_find_middle_candidate_score", &test_for_find_middle_candidate_score);
    m.def("test_for_find_middle_candidate_index", &test_for_find_middle_candidate_index);
    
    
    
    

    
}