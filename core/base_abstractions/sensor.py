# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) Allen Institute for AI
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from abc import abstractmethod, ABC
from collections import OrderedDict
from typing import (
    Generic,
    Dict,
    Any,
    Optional,
    TYPE_CHECKING,
    TypeVar,
    Sequence,
    cast,
    Tuple,
)

import PIL
import gym
import copy
import numpy as np
import torch

import core.base_abstractions.rgb_sensor_degradations as degradations

from gym.spaces import Dict as SpaceDict
from torch import nn
from torchvision import transforms, models

from core.base_abstractions.misc import EnvType
from utils.misc_utils import prepare_locals_for_super
from utils.model_utils import Flatten
from utils.tensor_utils import ScaleBothSides

if TYPE_CHECKING:
    from core.base_abstractions.task import SubTaskType
else:
    SubTaskType = TypeVar("SubTaskType", bound="Task")


class Sensor(Generic[EnvType, SubTaskType]):
    """Represents a sensor that provides data from the environment to agent.
    The user of this class needs to implement the get_observation method and
    the user is also required to set the below attributes:

    # Attributes

    uuid : universally unique id.
    observation_space : ``gym.Space`` object corresponding to observation of
        sensor.
    """

    uuid: str
    observation_space: gym.Space

    def __init__(self, uuid: str, observation_space: gym.Space, **kwargs: Any) -> None:
        self.uuid = uuid
        self.observation_space = observation_space

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        """Returns observations from the environment (or task).

        # Parameters

        env : The environment the sensor is used upon.
        task : (Optionally) a Task from which the sensor should get data.

        # Returns

        Current observation for Sensor.
        """
        raise NotImplementedError()


class SensorSuite(Generic[EnvType]):
    """Represents a set of sensors, with each sensor being identified through a
    unique id.

    # Attributes

    sensors: list containing sensors for the environment, uuid of each
        sensor must be unique.
    """

    sensors: Dict[str, Sensor[EnvType, Any]]
    observation_spaces: SpaceDict

    def __init__(self, sensors: Sequence[Sensor]) -> None:
        """Initializer.

        # Parameters

        param sensors: the sensors that will be included in the suite.
        """
        self.sensors = OrderedDict()
        spaces: OrderedDict[str, gym.Space] = OrderedDict()
        for sensor in sensors:
            assert (
                sensor.uuid not in self.sensors
            ), "'{}' is duplicated sensor uuid".format(sensor.uuid)
            self.sensors[sensor.uuid] = sensor
            spaces[sensor.uuid] = sensor.observation_space
        self.observation_spaces = SpaceDict(spaces=spaces)

    def get(self, uuid: str) -> Sensor:
        """Return sensor with the given `uuid`.

        # Parameters

        uuid : The unique id of the sensor

        # Returns

        The sensor with unique id `uuid`.
        """
        return self.sensors[uuid]

    def get_observations(
        self, env: EnvType, task: Optional[SubTaskType], **kwargs: Any
    ) -> Dict[str, Any]:
        """Get all observations corresponding to the sensors in the suite.

        # Parameters

        env : The environment from which to get the observation.
        task : (Optionally) the task from which to get the observation.

        # Returns

        Data from all sensors packaged inside a Dict.
        """
        return {
            uuid: sensor.get_observation(env=env, task=task, **kwargs)  # type: ignore
            for uuid, sensor in self.sensors.items()
        }


class ExpertActionSensor(Sensor[EnvType, SubTaskType]):
    def __init__(
        self,
        nactions: int,
        uuid: str = "expert_action",
        expert_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        self.nactions = nactions
        self.expert_args: Dict[str, Any] = expert_args or {}

        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Tuple:
        """The observation space of the expert action sensor.

        Will equal `gym.spaces.Tuple(gym.spaces.Discrete(num actions in
        task), gym.spaces.Discrete(2))` where the first entry of the
        tuple is the expert action index and the second equals 0 if and
        only if the expert failed to generate a true expert action. The
        value `num actions in task` should be in `config["nactions"]`
        """
        return gym.spaces.Tuple(
            (gym.spaces.Discrete(self.nactions), gym.spaces.Discrete(2))
        )

    def get_observation(
        self, env: EnvType, task: SubTaskType, *args: Any, **kwargs: Any
    ) -> Any:
        # If the task is completed, we needn't (perhaps can't) find the expert
        # action from the (current) terminal state.
        if task.is_done():
            return np.array([-1, False], dtype=np.int64)
        action, expert_was_successful = task.query_expert(**self.expert_args)
        assert isinstance(action, int), (
            "In expert action sensor, `task.query_expert()` "
            "did not return an integer action."
        )
        return np.array([action, expert_was_successful], dtype=np.int64)


class ExpertPolicySensor(Sensor[EnvType, SubTaskType]):
    def __init__(
        self,
        nactions: int,
        uuid: str = "expert_policy",
        expert_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        self.nactions = nactions
        self.expert_args: Dict[str, Any] = expert_args or {}

        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Tuple:
        """The observation space of the expert action sensor.

        Will equal `gym.spaces.Tuple(gym.spaces.Box(num actions in
        task), gym.spaces.Discrete(2))` where the first entry of the
        tuple is the expert policy and the second equals 0 if and only
        if the expert failed to generate a true expert action. The value
        `num actions in task` should be in `config["nactions"]`
        """
        return gym.spaces.Tuple(
            (
                gym.spaces.Box(
                    low=np.float32(0.0), high=np.float32(1.0), shape=(self.nactions,),
                ),
                gym.spaces.Discrete(2),
            )
        )

    def get_observation(
        self, env: EnvType, task: SubTaskType, *args: Any, **kwargs: Any
    ) -> Any:
        policy, expert_was_successful = task.query_expert(**self.expert_args)
        assert isinstance(policy, np.ndarray) and policy.shape == (self.nactions,), (
            "In expert action sensor, `task.query_expert()` "
            "did not return a valid numpy array."
        )
        return np.array(
            np.concatenate((policy, [expert_was_successful]), axis=-1), dtype=np.float32
        )


class RotationSensor(Sensor[EnvType, SubTaskType]):
    def __init__(self, uuid: str = "rot_label", **kwargs: Any):
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(4)

    def get_observation(
        self, env: EnvType, task: SubTaskType, *args: Any, **kwargs: Any
    ) -> Any:
        return 0


class SeparateRotatedVisionSensor(Sensor[EnvType, SubTaskType]):
    def __init__(
        self,
        mean: Optional[np.ndarray] = np.array(
            [[[0.485, 0.456, 0.406]]], dtype=np.float32
        ),
        stdev: Optional[np.ndarray] = np.array(
            [[[0.229, 0.224, 0.225]]], dtype=np.float32
        ),
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "sep_rot_rgb",
        output_shape: Optional[Tuple[int, ...]] = None,
        output_channels: int = 3,
        unnormalized_infimum: float = 0.0,
        unnormalized_supremum: float = 1.0,
        **kwargs: Any
    ):
        self._norm_means = mean
        self._norm_sds = stdev
        assert (self._norm_means is None) == (self._norm_sds is None), (
            "In SeparateRotatedVisionSensor's config, "
            "either both mean/stdev must be None or neither."
        )
        self._should_normalize = self._norm_means is not None

        self._height = height
        self._width = width
        assert (self._width is None) == (self._height is None), (
            "In SeparateRotatedVisionSensor's config, "
            "either both height/width must be None or neither."
        )

        observation_space = self._get_observation_space(
            output_shape=output_shape,
            output_channels=output_channels,
            unnormalized_infimum=unnormalized_infimum,
            unnormalized_supremum=unnormalized_supremum,
        )

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(
        self,
        output_shape: Optional[Tuple[int, ...]],
        output_channels: Optional[int],
        unnormalized_infimum: float,
        unnormalized_supremum: float,
    ) -> gym.spaces.Box:
        assert output_shape is None or output_channels is None, (
            "In VisionSensor's config, "
            "only one of output_shape and output_channels can be not None."
        )

        shape: Optional[Tuple[int, ...]] = None
        if output_shape is not None:
            shape = output_shape
        elif self._height is not None and output_channels is not None:
            shape = (
                cast(int, self._height),
                cast(int, self._width),
                cast(int, output_channels),
            )

        if not self._should_normalize or shape is None or len(shape) == 1:
            return gym.spaces.Box(
                low=np.float32(unnormalized_infimum),
                high=np.float32(unnormalized_supremum),
                shape=shape,
            )
        else:
            out_shape = shape[:-1] + (1,)
            low = np.tile(
                (unnormalized_infimum - cast(np.ndarray, self._norm_means))
                / cast(np.ndarray, self._norm_sds),
                out_shape,
            )
            high = np.tile(
                (unnormalized_supremum - cast(np.ndarray, self._norm_means))
                / cast(np.ndarray, self._norm_sds),
                out_shape,
            )
            return gym.spaces.Box(low=np.float32(low), high=np.float32(high))

    def get_observation(
        self, env: EnvType, task: SubTaskType, *args: Any, **kwargs: Any
    ) -> Any:
        return 0


class VisionSensor(Sensor[EnvType, SubTaskType]):
    def __init__(
        self,
        mean: Optional[np.ndarray] = None,
        stdev: Optional[np.ndarray] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "vision",
        output_shape: Optional[Tuple[int, ...]] = None,
        output_channels: Optional[int] = None,
        unnormalized_infimum: float = -np.inf,
        unnormalized_supremum: float = np.inf,
        scale_first: bool = True,
        **kwargs: Any
    ):
        """Initializer.

        # Parameters

        config : The images will be normalized
            with means `config["mean"]` and standard deviations `config["stdev"]`. If both `config["height"]` and
            `config["width"]` are non-negative integers then
            the image returned from the environment will be rescaled to have
            `config["height"]` rows and `config["width"]` columns using bilinear sampling. The universally unique
            identifier will be set as `config["uuid"]`.
        args : Extra args. Currently unused.
        kwargs : Extra kwargs. Currently unused.
        """

        print("UUID is ", uuid)

        def f(x, k, default):
            return x[k] if k in x else default

        self._random_crop: Optional[bool] = f(kwargs, "random_crop", False)
        self._crop_height: Optional[int] = f(kwargs, "crop_height", None)
        self._crop_width: Optional[int] = f(kwargs, "crop_width", None)
        self._jitter: Optional[bool] = f(kwargs, "color_jitter", False)
        self._gnoise: Optional[bool] = f(kwargs, "gaussian_noise", False)
        self._gblur: Optional[bool] = f(kwargs, "gaussian_blur", False)
        self._tshift: Optional[bool] = f(kwargs, "random_translate", False)
        self._daug_mode: Optional[bool] = f(kwargs, "data_augmentation_mode", False)

        # Parse corruption details
        # Additional inputs are
        # - a list of corruptions
        # - a list of severities
        self._corruptions = f(kwargs, "corruptions", None)
        self._severities = f(kwargs, "severities", None)

        print("Applied corruptions are ")
        print(self._corruptions)
        print(self._severities)

        print("Random Crop state ", self._random_crop)
        print("Color Jitter state ", self._jitter)
        # print("Gaussian Noise state ", self._gnoise)
        # print("Gaussian Blur state ", self._gblur)
        print("Random Translate ", self._tshift)

        # Whether to rotate the observation or not
        self._sep_rotate: bool = f(kwargs, "sep_rotate", False)

        self._norm_means = mean
        self._norm_sds = stdev
        assert (self._norm_means is None) == (self._norm_sds is None), (
            "In VisionSensor's config, "
            "either both mean/stdev must be None or neither."
        )
        self._should_normalize = self._norm_means is not None

        self._height = height
        self._width = width
        assert (self._width is None) == (self._height is None), (
            "In VisionSensor's config, "
            "either both height/width must be None or neither."
        )

        self._scale_first = scale_first

        self.scaler: Optional[ScaleBothSides] = None
        if self._width is not None:
            self.scaler = ScaleBothSides(
                width=cast(int, self._width), height=cast(int, self._height)
            )

        # Data augmentation options
        self._random_cropper = (
            None
            if not self._random_crop
            else transforms.RandomCrop((self._crop_height, self._crop_width))
        )

        self._color_jitter = (
            None if not self._jitter else transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
        )

        # # Operate maximally till severity = 2 equivalent of skimage gaussian blur
        # self._gaussian_blur = (
        #     None
        #     if not self._gblur
        #     else transforms.GaussianBlur((5, 5), sigma=(0.1, 2.0))
        # )

        self._random_translate = (
            None
            if not self._tshift
            else transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))
        )

        self.to_pil = transforms.ToPILImage()  # assumes mode="RGB" for 3 channels

        observation_space = self._get_observation_space(
            output_shape=output_shape,
            output_channels=output_channels,
            unnormalized_infimum=unnormalized_infimum,
            unnormalized_supremum=unnormalized_supremum,
        )

        assert int(PIL.__version__.split(".")[0]) < 7, (
            "Pillow version >=7.0.0 is very broken, please downgrade" "to version 6.2.1"
        )

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(
        self,
        output_shape: Optional[Tuple[int, ...]],
        output_channels: Optional[int],
        unnormalized_infimum: float,
        unnormalized_supremum: float,
    ) -> gym.spaces.Box:
        assert output_shape is None or output_channels is None, (
            "In VisionSensor's config, "
            "only one of output_shape and output_channels can be not None."
        )

        shape: Optional[Tuple[int, ...]] = None
        if output_shape is not None:
            shape = output_shape
        elif self._height is not None and output_channels is not None:
            shape = (
                cast(int, self._height),
                cast(int, self._width),
                cast(int, output_channels),
            )

        if not self._should_normalize or shape is None or len(shape) == 1:
            return gym.spaces.Box(
                low=np.float32(unnormalized_infimum),
                high=np.float32(unnormalized_supremum),
                shape=shape,
            )
        else:
            out_shape = shape[:-1] + (1,)
            low = np.tile(
                (unnormalized_infimum - cast(np.ndarray, self._norm_means))
                / cast(np.ndarray, self._norm_sds),
                out_shape,
            )
            high = np.tile(
                (unnormalized_supremum - cast(np.ndarray, self._norm_means))
                / cast(np.ndarray, self._norm_sds),
                out_shape,
            )
            return gym.spaces.Box(low=np.float32(low), high=np.float32(high))

    @property
    def height(self) -> Optional[int]:
        """Height that input image will be rescale to have.

        # Returns

        The height as a non-negative integer or `None` if no rescaling is done.
        """
        return self._height

    @property
    def width(self) -> Optional[int]:
        """Width that input image will be rescale to have.

        # Returns

        The width as a non-negative integer or `None` if no rescaling is done.
        """
        return self._width

    @abstractmethod
    def frame_from_env(self, env: EnvType) -> np.ndarray:
        raise NotImplementedError

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        im = self.frame_from_env(env)
        assert (im.shape[-1] == 1 and im.dtype == np.float32) or (
            im.shape[-1] == 3 and im.dtype == np.uint8
        ), (
            "Input frame must either have 3 channels and be of"
            " type np.uint8 or have one channel and be of type np.float32"
        )

        # Apply sequence of corruptions to
        # the RGB frames
        if self._corruptions is not None:
            im = degradations.apply_corruption_sequence(
                np.array(im), self._corruptions, self._severities
            )

        # if self._daug_mode:
        #     im = np.array(im)
        #     if self._random_crop:
        #         im = degradations.random_crop(im, (self._crop_height, self._crop_width))
        #     if self._gnoise:
        #         im = degradations.gaussian_noise(im, 2)
        #     if self._gblur:
        #         im = degradations.gaussian_blur(im, 2)
        #     if self._tshift:
        #         im = degradations.random_translate(im, 600, 600)

        #     if im.dtype in [np.float32, np.float64]:
        #         im = im.astype(np.uint8)

        #     im = degradations.apply_corruption_sequence(
        #         np.array(im), ["Gaussian Blur", "Gaussian Noise"], [2, 2],
        #     )

        if self._tshift:
            if isinstance(im, np.ndarray):
                im = self.to_pil(im)
            im = self._random_translate(im)

        # Random Crop Image
        if self._random_crop:
            if isinstance(im, np.ndarray):
                im = self.to_pil(im)
            im = self._random_cropper(im)

        # Color Jitter
        if self._jitter:
            if isinstance(im, np.ndarray):
                im = self.to_pil(im)
            im = self._color_jitter(im)

        # if self._gblur:
        #     if isinstance(im, np.ndarray):
        #         im = self.to_pil(im)
        #     im = self._gaussian_blur(im)

        if self._sep_rotate:
            rot_im = copy.deepcopy(im)

        if self._sep_rotate:
            if not isinstance(rot_im, np.ndarray):
                rot_im = np.array(im)
            rot_im, rot_label = degradations.rotate_single(rot_im)

        if self._scale_first:
            if not isinstance(im, np.ndarray):
                shape_condition = im.size[:2] != (self._height, self._width)
            else:
                shape_condition = im.shape[:2] != (self._height, self._width)
                im = self.to_pil(im)
            if self.scaler is not None and shape_condition:
                im = np.array(self.scaler(im), dtype=np.uint8)  # hwc

            if self._sep_rotate:
                if not isinstance(rot_im, np.ndarray):
                    shape_condition = rot_im.size[:2] != (self._height, self._width)
                else:
                    shape_condition = rot_im.shape[:2] != (self._height, self._width)
                    rot_im = self.to_pil(rot_im)
                if self.scaler is not None and shape_condition:
                    rot_im = np.array(self.scaler(rot_im), dtype=np.uint8)  # hwc

        # Original
        # if self._scale_first:
        #     if self.scaler is not None and im.shape[:2] != (self._height, self._width):
        #         im = np.array(self.scaler(self.to_pil(im)), dtype=im.dtype)  # hwc

        assert im.dtype in [np.uint8, np.float32]

        if self._sep_rotate:
            assert rot_im.dtype in [np.uint8, np.float32]

        if im.dtype == np.uint8:
            im = im.astype(np.float32) / 255.0

        if self._sep_rotate:
            if rot_im.dtype == np.uint8:
                rot_im = rot_im.astype(np.float32) / 255.0

        if self._should_normalize:
            im -= self._norm_means
            im /= self._norm_sds

        if self._sep_rotate:
            if self._should_normalize:
                rot_im -= self._norm_means
                rot_im /= self._norm_sds

        if not self._scale_first:  # Fix to be covered later
            if self.scaler is not None and im.shape[:2] != (self._height, self._width):
                im = np.array(self.scaler(self.to_pil(im)), dtype=np.float32)  # hwc

            if self._sep_rotate:
                if self.scaler is not None and rot_im.shape[:2] != (
                    self._height,
                    self._width,
                ):
                    rot_im = np.array(
                        self.scaler(self.to_pil(rot_im)), dtype=np.float32
                    )  # hwc

        if self._sep_rotate:
            return (im, rot_im, rot_label)
        else:
            return im


class RGBSensor(VisionSensor[EnvType, SubTaskType], ABC):
    def __init__(
        self,
        use_resnet_normalization: bool = False,
        mean: Optional[np.ndarray] = np.array(
            [[[0.485, 0.456, 0.406]]], dtype=np.float32
        ),
        stdev: Optional[np.ndarray] = np.array(
            [[[0.229, 0.224, 0.225]]], dtype=np.float32
        ),
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "rgb",
        output_shape: Optional[Tuple[int, ...]] = None,
        output_channels: int = 3,
        unnormalized_infimum: float = 0.0,
        unnormalized_supremum: float = 1.0,
        scale_first: bool = True,
        **kwargs: Any
    ):
        """Initializer.

        # Parameters

        config : If `config["use_resnet_normalization"]` is `True` then the RGB images will be normalized
            with means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]` (i.e. using the standard
            resnet normalization). If both `config["height"]` and `config["width"]` are non-negative integers then
            the RGB image returned from the environment will be rescaled to have shape
            (config["height"], config["width"], 3) using bilinear sampling.
        args : Extra args. Currently unused.
        kwargs : Extra kwargs. Currently unused.
        """

        if not use_resnet_normalization:
            mean, stdev = None, None

        super().__init__(**prepare_locals_for_super(locals()))


class DepthSensor(VisionSensor[EnvType, SubTaskType], ABC):
    def __init__(
        self,
        use_normalization: bool = False,
        mean: Optional[np.ndarray] = np.array([[0.5]], dtype=np.float32),
        stdev: Optional[np.ndarray] = np.array([[0.25]], dtype=np.float32),
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "depth",
        output_shape: Optional[Tuple[int, ...]] = None,
        output_channels: int = 1,
        unnormalized_infimum: float = 0.0,
        unnormalized_supremum: float = 5.0,
        scale_first: bool = True,
        **kwargs: Any
    ):
        """Initializer.

        # Parameters

        config : If `config["use_normalization"]` is `True` then the depth images will be normalized
            with mean 0.5 and standard deviation 0.25. If both `config["height"]` and `config["width"]` are
            non-negative integers then the depth image returned from the environment will be rescaled to have shape
            (config["height"], config["width"]) using bilinear sampling.
        args : Extra args. Currently unused.
        kwargs : Extra kwargs. Currently unused.
        """

        if not use_normalization:
            mean, stdev = None, None

        super().__init__(**prepare_locals_for_super(locals()))

    # def get_observation(  # type: ignore
    #     self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    # ) -> Any:
    #     depth = super().get_observation(env, task, *args, **kwargs)
    #     depth = np.expand_dims(depth, 2)

    #     return depth


class ResNetSensor(VisionSensor[EnvType, SubTaskType], ABC):
    def __init__(
        self,
        mean: Optional[np.ndarray] = None,
        stdev: Optional[np.ndarray] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "resnet",
        output_shape: Optional[Tuple[int, ...]] = None,
        output_channels: Optional[int] = None,
        unnormalized_infimum: float = -np.inf,
        unnormalized_supremum: float = np.inf,
        scale_first: bool = True,
        **kwargs: Any
    ):
        self.to_tensor = transforms.ToTensor()

        self.resnet = nn.Sequential(
            *list(models.resnet50(pretrained=True).children())[:-1] + [Flatten()]
        ).eval()

        self.device: torch.device = torch.device("cpu")

        super().__init__(**prepare_locals_for_super(locals()))

    def to(self, device: torch.device) -> "ResNetSensor":
        """Moves sensor to specified device.

        # Parameters

        device : The device for the sensor.
        """
        self.device = device
        self.resnet = self.resnet.to(device)
        return self

    def observation_to_tensor(self, observation: Any) -> torch.Tensor:
        return self.to_tensor(observation)

    def get_observation(  # type: ignore
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        observation = super().get_observation(env, task, *args, **kwargs)

        input_tensor = (
            self.observation_to_tensor(observation).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            result = self.resnet(input_tensor).detach().cpu().numpy()

        return result


class RGBResNetSensor(ResNetSensor[EnvType, SubTaskType], ABC):
    def __init__(
        self,
        use_resnet_normalization: bool = True,
        mean: Optional[np.ndarray] = np.array(
            [[[0.485, 0.456, 0.406]]], dtype=np.float32
        ),
        stdev: Optional[np.ndarray] = np.array(
            [[[0.229, 0.224, 0.225]]], dtype=np.float32
        ),
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "rgbresnet",
        output_shape: Optional[Tuple[int, ...]] = (2048,),
        output_channels: Optional[int] = None,
        unnormalized_infimum: float = -np.inf,
        unnormalized_supremum: float = np.inf,
        scale_first: bool = True,
        **kwargs: Any
    ):
        """Initializer.

        # Parameters

        config : If `config["use_resnet_normalization"]` is `True` then the RGB images will be normalized
            with means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]` (i.e. using the standard
            resnet normalization). If both `config["height"]` and `config["width"]` are non-negative integers then
            the RGB image returned from the environment will be rescaled to have shape
            (config["height"], config["width"], 3) using bilinear sampling before being fed to a ResNet-50 and
            extracting the flattened 2048-dimensional output embedding.
        args : Extra args. Currently unused.
        kwargs : Extra kwargs. Currently unused.
        """
        if not use_resnet_normalization:
            mean, stdev = None, None

        super().__init__(**prepare_locals_for_super(locals()))


class DepthResNetSensor(ResNetSensor[EnvType, SubTaskType], ABC):
    def __init__(
        self,
        use_normalization: bool = False,
        mean: Optional[np.ndarray] = np.array([[0.5]], dtype=np.float32),
        stdev: Optional[np.ndarray] = np.array([[0.25]], dtype=np.float32),
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "depthresnet",
        output_shape: Optional[Tuple[int, ...]] = (2048,),
        output_channels: Optional[int] = None,
        unnormalized_infimum: float = -np.inf,
        unnormalized_supremum: float = np.inf,
        scale_first: bool = True,
        **kwargs: Any
    ):
        """Initializer.

        # Parameters

        config : If `config["use_normalization"]` is `True` then the depth images will be normalized
            with mean 0.5 and standard deviation 0.25. If both `config["height"]` and `config["width"]` are
            non-negative integers then the depth image returned from the environment will be rescaled to have shape
            (config["height"], config["width"], 1) using bilinear sampling before being replicated to fill in three
            channels to feed a ResNet-50 and finally extract the flattened 2048-dimensional output embedding.
        args : Extra args. Currently unused.
        kwargs : Extra kwargs. Currently unused.
        """

        if not use_normalization:
            mean, stdev = None, None

        super().__init__(**prepare_locals_for_super(locals()))

    def observation_to_tensor(self, depth: Any) -> torch.Tensor:
        depth = super().observation_to_tensor(depth).squeeze()
        return torch.stack([depth] * 3, dim=0)
