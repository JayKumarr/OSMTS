


class Dataset:
    def __init__(self, total_instances, minimum_dimension_for_doc):
        self.X = [None for a in range(0,total_instances)]
        self.Y = [None for a in range(0,total_instances)]

        self. counter_index = 0

        self.stat_lowest_dimension_value = -1
        self.stat_highest_dimension_value = -1

        self.mimum_dimension = minimum_dimension_for_doc


    def add_instance(self, x, y):
        temp_x=list(x)
        if ( temp_x.__len__()<self.mimum_dimension ):
            del self.X[-1]
            del self.Y[-1]
            return self.X.__len__()
        temp_y=list(y)
        self.X[self.counter_index] = temp_x
        self.Y[self.counter_index] = temp_y

        self.counter_index = self.counter_index+1

        if temp_x.__len__()<self.stat_lowest_dimension_value:
            self.stat_lowest_dimension_value = temp_x.__len__()
        if temp_x.__len__()>self.stat_highest_dimension_value:
            self.stat_highest_dimension_value = temp_x.__len__()

        return self.X.__len__()



