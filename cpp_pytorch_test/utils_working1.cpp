#include <torch/extension.h>
#include <torch/torch.h>
#include <iostream>

using namespace std;



at::Tensor trilinear_fw_test(
    const torch::Tensor feats,
    const torch::Tensor points
)
{
    //cout<<feats.shape;
    return feats.size();


}


struct sorted_point_cloud
{
    torch::Tensor projected_values,
    torch::Tensor order   
}


sorted_point_cloud project_an_sort(torch::Tensor xyz) 
{
    cout<<xyz.shape;
}

torch::Tensor farthest_point_sample(
    torch::Tensor xyz,
    torch::Tensor npoint
)
{

}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_fw_test", &trilinear_fw_test);
}
