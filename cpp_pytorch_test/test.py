import torch
import cppcuda_tutorial
import numpy as np


#print(cppcuda_tutorial.trilinear_fw_test(torch.ones(2,3), torch.zeros(1,2)))




A = torch.tensor(([30,4,5],[2,-3,4]))
#print(A[0,1])
#print(cppcuda_tutorial.getFirstElement(A))
#print(A)
A = torch.rand(500, 2014, 3)

#A = torch.tensor(A)
#print(A.shape)
#A = torch.ones((500, 1024, 3))
#print(cppcuda_tutorial.test_working_with_queue())
#A = torch.tensor([4,3,-9,9,34,-90,23,86,12,1]).sort()
A = torch.tensor([[-90,  -9,   1,   3,   4,   9,  12,  23,  34,  100]])

print(cppcuda_tutorial.test_for_binary_search_score(A, 0, 9, -10))
print(cppcuda_tutorial.test_for_binary_search_index(A, 0, 9, -8.02))
print(cppcuda_tutorial.test_for_find_middle_candidate_score(A, 0, 9))
print(cppcuda_tutorial.test_for_find_middle_candidate_index(A, 0, 9))
#print(cppcuda_tutorial.project_and_sort(A).order)
#print(cppcuda_tutorial.farthest_point_sampling(A, A))

#print("modified to: ")
#print(cppcuda_tutorial.modifyTensor(A))

