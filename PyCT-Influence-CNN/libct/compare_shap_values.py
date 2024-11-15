
from __future__ import annotations
import sys

import functools
import shap
from keras import Model
from keras.layers import Layer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import InputLayer
from typing import Tuple
import numpy as np

from libct.constraint import Constraint

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable
    class PositionedConstraint(Tuple[Constraint, Tuple[int, Tuple[int, ...]]]):
        pass





def pop_last_constraint(positioned_constraints: list[PositionedConstraint]) -> Constraint:
    return positioned_constraints.pop()[0]


def pop_first_constraint(positioned_constraints: list[PositionedConstraint]) -> Constraint:
    return positioned_constraints.pop(0)[0]


def pop_the_most_important_constraint(
        positioned_constraints: list[PositionedConstraint],
        compare: Callable[[PositionedConstraint, PositionedConstraint], int]
) -> Constraint:
    positioned_constraints.sort(key=functools.cmp_to_key(compare))
    return pop_last_constraint(positioned_constraints)


class ShapValuesComparator:
    def __init__(self, model: Model, background_dataset, input) -> None:
        self.model = model
        self.background_dataset = background_dataset
        self.input = input
        self.shap_values: dict[str, float] = dict()
        self.calculate_shap_values_for_all_neurons()

    def compare(self, positioned_constraint_1: PositionedConstraint, positioned_constraint_2: PositionedConstraint) -> float:
        constraint_1, (row_number_1, indices_1) = positioned_constraint_1
        constraint_2, (row_number_2, indices_2) = positioned_constraint_2
        # if row_number_1 > row_number_2: return -1
        # if row_number_1 < row_number_2: return 1
        return self.get_shap_value(row_number_1, indices_1) - self.get_shap_value(row_number_2, indices_2)

    def calculate_shap_values_for_all_neurons(self) -> None:
        trimmed_model = self.model
        transformed_background = self.background_dataset
        transformed_input = self.input
        print("------trimmed_model input shape ------", trimmed_model.input_shape)
        print("------transformed_input------", transformed_input.shape)
        print("------transformed_background------", transformed_background.shape)
        number_of_layers = len(self.model.layers)
        for layer_number in range(0, number_of_layers):
            print("------layer_number------", layer_number)
            self.calculate_shap_values(
                trimmed_model, transformed_background, transformed_input, layer_number)
            transformed_input = ShapValuesComparator.apply_one_layer( trimmed_model, transformed_input)
            transformed_background = ShapValuesComparator.apply_one_layer_to_dataset( trimmed_model, transformed_background)
            if layer_number == number_of_layers - 1: break
            trimmed_model = ShapValuesComparator.without_first_layer(trimmed_model)
        # print("------shap_values------", self.shap_values.keys())
        # exit(0)
# input, layer 1, layer 2, layer 3, layer 4, layer 5
# -1,    0,        1,       2,       3,       4
#                                             inf
# if-then-else + concolic -> constraint -> neuron -> shap value

    @staticmethod
    def without_first_layer(original_model: Sequential | Model) -> Sequential | Model:
        if isinstance(original_model, Sequential):
            new_model = Sequential()
            for layer in original_model.layers[1:]:
                new_model.add(layer)
            new_model.build(original_model.layers[1].input_shape)
        elif isinstance(original_model, Model):
            # Get the output of the second layer
            new_input = original_model.layers[1].output
            # Get the output of the last layer
            new_output = original_model.layers[-1].output
            new_model = Model(inputs=new_input, outputs=new_output)
        return new_model

    @staticmethod
    def without_first_layer_fast(original_model: Sequential | Model) -> Sequential | Model:
        if isinstance(original_model, Sequential):
            new_model = Sequential(original_model.layers[1:])
            new_model.build(original_model.layers[1].input_shape)
        elif isinstance(original_model, Model):
            # Get the output of the second layer
            new_input = original_model.layers[1].output
            # Get the output of the last layer
            new_output = original_model.layers[-1].output
            new_model = Model(inputs=new_input, outputs=new_output)
        return new_model

    @staticmethod
    def apply_one_layer_to_dataset( original_model: Sequential | Model, dataset: np.ndarray) -> np.ndarray:
        model_with_only_first_layer = ShapValuesComparator.get_model_with_only_first_layer(
            original_model)
        model_with_only_first_layer.build(original_model.layers[0].input_shape)
        new_dataset = model_with_only_first_layer.predict(dataset)
        # new_dataset = np.array([model_with_only_first_layer.predict(data) for data in dataset ])
        return new_dataset
    
    @staticmethod
    def apply_one_layer( original_model: Sequential | Model, original_input: np.ndarray) -> np.ndarray:
        return ShapValuesComparator.get_model_with_only_first_layer(original_model).predict(original_input)

    @staticmethod
    def get_model_with_only_first_layer(original_model: Sequential | Model) -> Sequential | Model:
        if isinstance(original_model, Sequential):
            model_with_only_first_layer = Sequential(original_model.layers[:1])
        elif isinstance(original_model, Model):
            model_with_only_first_layer = Model(
                inputs=original_model.input, outputs=original_model.layers[0].output)
        return model_with_only_first_layer
    
    def calculate_shap_values(self, model, background_dataset, input, layer_number: int) -> None:
        modelInputAndOutput = (model.layers[0].input, model.layers[-1].output)
        # shap_values = shap.DeepExplainer(
        #     modelInputAndOutput, input).shap_values(input)
        #background_dataset = background_dataset.reshape((100, -1))
        # print("------background_dataset------", background_dataset)
        # print("------input------", input.shape)
        # print("------background_dataset.shape------", background_dataset.shape)
      
        explainer = shap.DeepExplainer(model, background_dataset) # background_dataset.shape = (100, 28, 28)
        shap_values = explainer.shap_values(input) # input.shape = (1, 28, 28)
        for indices, shap_value in np.ndenumerate(shap_values):
            # print("------indices------", indices)
            indices = indices[2:] # remove the first two dimensions
            # print("------indices[2:]------", indices)
            self.shap_values[ShapValuesComparator.get_position_key(
                layer_number - 1, indices)] = shap_value
        
    def get_shap_value(self, layer_number: int, indices: tuple[int, ...]) -> float:
        # print("------self.shap_values positionkeys------", ShapValuesComparator.get_position_key(layer_number, indices))
        # print("------self.shap_values------")
        # for key, value in list(self.shap_values.items())[:10]:
        #     print(key, value)
        # print("---layer_number---", layer_number)
        # print("---indices---", indices)
        
        number_of_layers = len(self.model.layers)
        if layer_number == number_of_layers - 1:
            return float('inf')
        return self.shap_values[ShapValuesComparator.get_position_key(layer_number, indices)]

    @staticmethod
    def get_position_key(layer_number: int, indices: tuple[int, ...]) -> str:
        key = str(layer_number)
        # if layer_number == 1:
        #     print("------key------", key)
        for i in indices:
            key += "_" + str(i)
        # if layer_number == 1:
        #     print("------key after + indices------", key)
        return key
    def print_shap_values(self) -> None:
      for key in self.shap_values:
        print(key, self.shap_values[key])

    

# usage:
# positioned_constraints: List[PositionedConstraint] = ...
# constraint: Constraint = pop_the_most_important_constraint(positioned_constraints, ShapValuesComparator().compare)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
input_shape = (28, 28, 1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 將標籤進行 one-hot 編碼
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

class ShapValuesComparatorTester:
    def __init__(self):
        # construct a model with three Layers
        self.input_data = np.random.rand(1000, 10)
        self.model = Sequential()
        self.model.add(Dense(units=32, activation='relu', input_shape=(10,)))
        self.model.add(Dense(units=1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()
        self.labels = np.random.randint(2, size=(1000, 1))
        self.model.fit(self.input_data, self.labels, epochs=10, batch_size=32)
        self.background_dataset = self.input_data[np.random.choice(self.input_data.shape[0], 100, replace=False)]
        #self.background_dataset = self.background_dataset.reshape((100, -1))
        # create an input of floats between 0 and 1
        self.input = self.input_data[:1]
        # create a comparator
        self.runtests()

    def runtests(self):
        #self.test_get_position_key()
        # self.test_without_first_layer()
        # self.test_apply_one_layer()
        # self.test_apply_one_layer_to_dataset()
        # self.test_get_model_with_only_first_layer()
        self.comparator = ShapValuesComparator(
            self.model, self.background_dataset, self.input)
        print(self.comparator.shap_values)
        # self.test_calculate_shap_values()
        # self.test_calculate_shap_values_for_all_neurons()
        # self.test_get_shap_value()
        # self.test_compare()

    def test_get_position_key(self):
        assert ShapValuesComparator.get_position_key(1, (1, 2)) == "1_1_2"
        assert ShapValuesComparator.get_position_key(1, (1, 2, 3)) == "1_1_2_3"
        assert ShapValuesComparator.get_position_key(1, ()) == "1"


    def test_without_first_layer(self):
        # self.model.summary()
        new_model = ShapValuesComparator.without_first_layer(self.model)
        # new_model.summary()
        assert len(new_model.layers) == 2
        assert new_model.layers[0].input_shape == (None, 64)
        assert new_model.layers[1].input_shape == (None, 10)
        assert new_model.layers[0].output_shape == (None, 10)
        assert new_model.layers[0].get_config()['activation'] == 'relu'
        assert new_model.layers[1].get_config()['activation'] == 'softmax'

    def test_apply_one_layer(self):
        print("2-1")
        assert self.input.shape == (1, 64)
        print("2-2")
        assert self.model.layers[0].input_shape == (None, 64)
        print("2-3")
        assert self.model.layers[0].output_shape == (None, 32)
        print("2-4")
        output = ShapValuesComparator.apply_one_layer(self.model, self.input)
        print("2-5")
        assert output.shape == (1, 32)


    def test_apply_one_layer_to_dataset(self):
        assert self.background_dataset.shape == (100, 1, 64)
        assert self.model.layers[0].input_shape == (None, 64)
        assert self.model.layers[0].output_shape == (None, 32)
        print("3-1")
        output = ShapValuesComparator.apply_one_layer_to_dataset(
            self.model, self.background_dataset)
        assert output.shape == (100, 1, 32)

    def test_get_model_with_only_first_layer(self):
        model_with_only_first_layer = ShapValuesComparator.get_model_with_only_first_layer(
            self.model)
        assert len(model_with_only_first_layer.layers) == 1
        assert model_with_only_first_layer.layers[0].input_shape == self.model.layers[0].input_shape
        assert model_with_only_first_layer.layers[0].output_shape == self.model.layers[0].output_shape
        assert model_with_only_first_layer.layers[0].activation == self.model.layers[0].activation

    def test_calculate_shap_values(self):
        self.comparator.calculate_shap_values(
            self.model, self.background_dataset, self.input, 0)
        self.print_shap_values()
        assert len(self.comparator.shap_values) >= 64
        assert '0_0_0_1' in self.comparator.shap_values
        assert '0_0_0_63' in self.comparator.shap_values
    def print_shap_values(self):
        for k, v in self.comparator.shap_values.items():
            print(k, v)

    def test_get_shap_value(self):
        self.comparator.calculate_shap_values(
            self.model, self.background_dataset, self.input, 1)
        assert self.comparator.get_shap_value(
            1, (0,0,0)) == self.comparator.shap_values['1_0_0_0']
        assert self.comparator.get_shap_value(
            1, (0,0,31)) == self.comparator.shap_values['1_0_0_31']

    def test_calculate_shap_values_for_all_neurons(self):
        assert len(self.comparator.shap_values) == 64 + 32 + 64 + 10
        assert self.comparator.shap_values.get('0_0_0_0') != None
        assert self.comparator.shap_values.get('0_0_0_63') != None
        assert self.comparator.shap_values.get('1_0_0_0') != None
        assert self.comparator.shap_values.get('1_0_0_31') != None
        assert self.comparator.shap_values.get('2_0_0_0') != None
        assert self.comparator.shap_values.get('2_0_0_63') != None
        assert self.comparator.shap_values.get('3_0_0_0') != None
        assert self.comparator.shap_values.get('3_0_0_9') != None

    def test_compare(self):
        mock_constraint_1 = Constraint(None, None)
        compare = self.comparator.compare
        positioned_constraint_1: PositionedConstraint = (
            mock_constraint_1, (1, (0,0,1)))
        positioned_constraint_2: PositionedConstraint = (
            mock_constraint_1, (1, (0,0,2)))
        if self.comparator.shap_values['1_0_0_1'] > self.comparator.shap_values['1_0_0_2']:
            assert compare(positioned_constraint_1,
                           positioned_constraint_2) > 0
        elif self.comparator.shap_values['1_0_0_1'] < self.comparator.shap_values['1_0_0_2']:
            assert compare(positioned_constraint_1,
                           positioned_constraint_2) < 0
        else:
            assert compare(positioned_constraint_1,
                           positioned_constraint_2) == 0

if __name__ == "__main__":
    ShapValuesComparatorTester()