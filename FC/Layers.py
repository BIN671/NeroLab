import numpy as np



class Dense:
    def __init__(self, size, function, previous_dense):
        self.size = size

        self.activate_f = np.vectorize(function.activate)
        self.derivative_f = np.vectorize(function.derivative)
        self.previous_dense = previous_dense
        self.weights_matrix = np.random.randn(size, previous_dense.size+1)
        self.own_matrix = np.empty((size, 1))
        self.previous_matrix = previous_dense.own_matrix
        self.own_result = np.empty((size, 1))
        self.previous_result = previous_dense.own_result


    def return_meaning(self, input_matrix):
        #print(input_matrix)
        prod_matrix = np.dot(self.weights_matrix, np.vstack([input_matrix, [[1]]]))
        #print(np.vstack([input_matrix, [[1]]]))
        #print(prod_matrix)
        self.own_matrix[:] = prod_matrix
        result_matrix = self.activate_f(prod_matrix)
        self.own_result[:] = result_matrix
        #print(self.weights_matrix)
        return self.next_dense.return_meaning(result_matrix)

    def get_wired(self, next_dense):
        self.next_dense = next_dense
        self.previous_dense.get_wired(self)

    def mutation(self, mutation_function):
        self.weights_matrix[:] = mutation_function(self.weights_matrix)
        self.next_dense.mutation(mutation_function)

    def cross(self, models_layers, n):
        self.weights_matrix = np.sum([model_layer.weights_matrix for model_layer in models_layers], axis=0) / n
        self.next_dense.cross([model_layer.next_dense for model_layer in models_layers], n)

    def learn(self, ERROR, loss, rate):
        A = self.previous_matrix
        W = self.weights_matrix
        O_M = self.own_matrix
        new_A = loss.grad_a_out(ERROR, W, O_M, self.derivative_f, rate)
        grad_W = loss.grad_w_out(ERROR, A, O_M, self.derivative_f, rate)
        self.weights_matrix = self.weights_matrix - grad_W
        return self.previous_dense.learn(new_A, loss, rate)




class InputLayer:
    def __init__(self, size):
        self.size = size
        self.own_matrix = np.empty((size, 1))
        #print(self.own_matrix)
        self.own_result = np.empty((size, 1))

    def return_meaning(self, data):
        self.own_matrix[:] = np.array(data).reshape(-1, 1)
        #print(self.own_matrix)
        return self.next_dense.return_meaning(self.own_matrix)

    def get_wired(self, next_dense):
        self.next_dense = next_dense
        print("компиляция завершена")

    def begin_mutation(self, mutation_function):
        self.next_dense.mutation(mutation_function)

    def begin_cross(self, models, n):
        self.next_dense.cross([model.input_layer.next_dense for model in models], n)

    def learn(self, ERROR, loss, rate):
        #print("обучение завершенно")
        pass


class OutputLayer(Dense):
    def get_wired(self):
        self.previous_dense.get_wired(self)

    def return_meaning(self, input_matrix):
        prod_matrix = np.dot(self.weights_matrix, np.vstack([input_matrix, [[1]]]))
        self.own_matrix[:] = prod_matrix
        result_matrix = self.activate_f(prod_matrix)
        self.own_result[:] = result_matrix
        #print(id(self.previous_matrix))
        #print(self.weights_matrix)
        #print(result_matrix)
        return result_matrix

    def mutation(self, mutation_function):
        self.weights_matrix[:] = mutation_function(self.weights_matrix)
        print("мутация завершена")

    def cross(self, models_layers, n):
        self.weights_matrix = np.sum([model_layer.weights_matrix for model_layer in models_layers], axis=0) / n
        print("cкрещивание завершено")

class Model:
    def __init__(self, input_layer, output_layer):
        self.input_layer = input_layer
        self.output_layer = output_layer
        #self.loss = loss

    def compile(self):
        self.output_layer.get_wired()

    def predict(self, data):
        return self.input_layer.return_meaning(data)

    def learn(self, X, Y, loss, rate = 0.01):
        """
        during_dense = self.output_layer
        result = self.predict(X)
        A = during_dense.previous_matrix
        W = during_dense.weights_matrix
        O_M = during_dense.own_matrix
        new_A = self.loss.grad_a_out(result, Y, W, O_M, during_dense.derivative_f, rate)
        grad_W = self.loss.grad_w_out(result, Y, A, O_M, during_dense.derivative_f, rate)
        during_dense.weights_matrix = during_dense.weights_matrix-grad_W
        return during_dense.previous_dense.learn(new_A)
        """
        Test = np.array(Y).reshape(-1, 1)
        D = self.predict(X)
        #print(f"ошибка {(D - Test)}")
        ERROR = loss.error1(D, Test)
        #print(ERROR)
        return self.output_layer.learn(ERROR, loss, rate)













    @staticmethod
    def mutate_model(model, mutation_function):
        mutation_function = np.vectorize(mutation_function, otypes=[float])
        model.input_layer.begin_mutation(mutation_function)


    @staticmethod
    def create_derived_model(models):
        n = len(models)
        model = copy.deepcopy(models[0])
        model.input_layer.begin_cross(models, n)
        return model

    @staticmethod
    def copy_model(model):
        return copy.deepcopy(model)