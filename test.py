nums = [5, 2, 3, 1]

enum_nums = list(enumerate(nums))
enum_nums.sort(key=lambda x: x[1])

sorted_nums = [x[1] for x in enum_nums]  
original_indices = [x[0] for x in enum_nums]

print(sorted_nums) # [1, 2, 3, 5]
print(original_indices) # [3, 1, 2, 0]
