
def sort_array(array):
    """
    Given an array of non-negative integers, return a copy of the given array after sorting,
    you will sort the given array in ascending order if the sum( first index value, last index value) is odd,
    or sort it in descending order if the sum( first index value, last index value) is even.

    Note:
    * don't change the given array.

    Examples:
    * sort_array([]) => []
    * sort_array([5]) => [5]
    * sort_array([2, 4, 3, 0, 1, 5]) => [0, 1, 2, 3, 4, 5]
    * sort_array([2, 4, 3, 0, 1, 5, 6]) => [6, 5, 4, 3, 2, 1, 0]
    """
    if len(array) == 0:
        return array
    if len(array) == 1:
        return array
    if len(array) == 2:
        if array[0] > array[1]:
            return [array[1], array[0]]
        else:
            return array
    if len(array) == 3:
        if array[0] > array[1]:
            if array[1] > array[2]:
                return [array[2], array[1], array[0]]
            else:
                return [array[1], array[2], array[0]]
        else:
            if array[1] > array[2]:
                return [array[2], array[0], array[1]]
            else:
                return [array[0], array[1], array[2]]
    if len(array) == 4:
        if array[0] > array[1]:
            if array[1] > array[2]:
                if array[2] > array[3]:
                    return [array[3], array[2], array[1], array[0]]
                else:
                    return [array[2], array[3], array[1], array[0]]
            else:
                if array[2] > array[3]:
                    return [array[3], array[2], array[0], array[1]]
                else:
                    return [array[2], array[3], array[0], array[1]]
        else:
            if array[1] > array[2]:
                if array[2] > array[3]:
                    return [array[3], array[2], array[0], array[1]]
                else:
                    return [array[2], array[3], array[0], array[1]]
            else:
                if array[2] > array[3]:
                    return [array[3], array[0], array[1], array[2]]
                else:
                    return [array[0], array[1], array[2], array[3]]
    if len(array) == 5:
        if array[0] > array[1]:
            if array[1] > array[2]:
                if array[2] > array[3]:
                    if array[3] > array[4]:
                        return [array[4], array[3], array[2], array[1], array[0]]
                    else:
                        return [array[3], array[4], array[2], array[1], array[0]]
                else:
                    if array[3] > array[4]:
                        return [array[4], array[3], array[0], array[1], array[2]]
                    else:
                        return [array[3], array[4], array[0], array[1], array[2]]
            else:
                if array[2] > array[3]:
                    if array[3] > array[4]:
                        return [array[4], array[3], array[0], array[1], array[2]]
                    else:
                        return [array[3], array[4], array[0], array[1], array[2]]
                else:
                    if array[3] > array[4]:
                        return [array[4], array[0], array[1], array[2], array[3]]
                    else:
                        return [array[0], array[1], array[2], array[3], array[4]]
        else:
            if array[1] > array[2]:
                if array[2] > array[3]:
                    if array[3] >