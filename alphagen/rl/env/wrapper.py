
from typing import Tuple, List
import gymnasium as gym
import numpy as np

from alphagen.config import *
from alphagen.data.tokens import *
from alphagen.models.alpha_pool import AlphaPoolBase, AlphaPool
from alphagen.rl.env.core import AlphaEnvCore

SIZE_NULL = 1
SIZE_OP = len(OPERATORS)
SIZE_FEATURE = len(FEATURES)
SIZE_DELTA_TIME = len(DELTA_TIMES)
SIZE_CONSTANT = len(CONSTANTS)
SIZE_SEP = 1

SIZE_ALL = SIZE_NULL + SIZE_OP + SIZE_FEATURE + SIZE_DELTA_TIME + SIZE_CONSTANT + SIZE_SEP
SIZE_ACTION = SIZE_ALL - SIZE_NULL

OFFSET_OP = SIZE_NULL
OFFSET_FEATURE = OFFSET_OP + SIZE_OP
OFFSET_DELTA_TIME = OFFSET_FEATURE + SIZE_FEATURE
OFFSET_CONSTANT = OFFSET_DELTA_TIME + SIZE_DELTA_TIME
OFFSET_SEP = OFFSET_CONSTANT + SIZE_CONSTANT


def action2token(action_raw: int) -> Token:
    action = action_raw + 1
    if action < OFFSET_OP:
        raise ValueError
    elif action < OFFSET_FEATURE:
        return OperatorToken(OPERATORS[action - OFFSET_OP])
    elif action < OFFSET_DELTA_TIME:
        return FeatureToken(FEATURES[action - OFFSET_FEATURE])
    elif action < OFFSET_CONSTANT:
        return DeltaTimeToken(DELTA_TIMES[action - OFFSET_DELTA_TIME])
    elif action < OFFSET_SEP:
        return ConstantToken(CONSTANTS[action - OFFSET_CONSTANT])
    elif action == OFFSET_SEP:
        return SequenceIndicatorToken(SequenceIndicatorType.SEP)
    else:
        assert False


def token2action(token: Token) -> int:
    if isinstance(token, OperatorToken):
        return OPERATORS.index(token.operator) + OFFSET_OP
    elif isinstance(token, FeatureToken):
        return FEATURES.index(token.feature) + OFFSET_FEATURE
    elif isinstance(token, DeltaTimeToken):
        return DELTA_TIMES.index(token.delta_time) + OFFSET_DELTA_TIME
    elif isinstance(token, ConstantToken):
        return CONSTANTS.index(token.constant) + OFFSET_CONSTANT
    elif isinstance(token, SequenceIndicatorToken):
        return OFFSET_SEP
    else:
        assert False


def get_operator_arity(action: int) -> int:
    action_offset = action + 1
    if action_offset < OFFSET_OP:
        return -1
    elif action_offset < OFFSET_FEATURE:
        op_idx = action_offset - OFFSET_OP
        operator = OPERATORS[op_idx]
        if hasattr(operator, 'n_args'):
            return operator.n_args
        if hasattr(operator, 'arity'):
            return operator.arity
        category = operator.category_type() if hasattr(operator, 'category_type') else None
        if category in ['rolling', 'pair_rolling']:
            return 1
        elif category in ['binary']:
            return 2
        return 1
    elif action_offset < OFFSET_DELTA_TIME:
        return 0
    elif action_offset < OFFSET_CONSTANT:
        return 0
    elif action_offset < OFFSET_SEP:
        return 0
    elif action_offset == OFFSET_SEP:
        return -1
    return -1


class DenseRewardCalculator:

    def __init__(self,
                 subtree_reward: float = 0.01,
                 complexity_reward: float = 0.005,
                 syntax_reward: float = 0.002,
                 diversity_reward: float = 0.003,
                 completion_bonus: float = 0.02,
                 target_depth: int = 5):
        self.subtree_reward = subtree_reward
        self.complexity_reward = complexity_reward
        self.syntax_reward = syntax_reward
        self.diversity_reward = diversity_reward
        self.completion_bonus = completion_bonus
        self.target_depth = target_depth
        self.reset()

    def reset(self):
        self.stack_depth = 0
        self.max_stack_depth = 0
        self.subtree_count = 0
        self.used_operators = set()
        self.expression_depth = 0
        self.step_count = 0

    def calculate_reward(self, action: int, is_valid: bool = True) -> Tuple[float, dict]:
        if not is_valid:
            return -0.01, {'type': 'invalid'}

        self.step_count += 1
        reward = 0.0
        components = {}

        arity = get_operator_arity(action)
        action_offset = action + 1

        reward += self.syntax_reward
        components['syntax'] = self.syntax_reward

        if arity == -1:
            if action_offset == OFFSET_SEP:
                if self.stack_depth == 1:
                    reward += self.completion_bonus
                    components['completion'] = self.completion_bonus
                    if self.expression_depth >= self.target_depth:
                        depth_bonus = self.completion_bonus
                        reward += depth_bonus
                        components['depth_bonus'] = depth_bonus

        elif arity == 0:
            self.stack_depth += 1
            self.max_stack_depth = max(self.max_stack_depth, self.stack_depth)

        elif arity == 1:
            if self.stack_depth >= 1:
                reward += self.subtree_reward
                components['subtree'] = self.subtree_reward
                self.subtree_count += 1
                self.expression_depth += 1

            op_idx = action_offset - OFFSET_OP
            if op_idx not in self.used_operators:
                self.used_operators.add(op_idx)
                reward += self.diversity_reward
                components['diversity'] = self.diversity_reward

        elif arity == 2:
            if self.stack_depth >= 2:
                subtree_r = self.subtree_reward * 1.5
                reward += subtree_r
                components['subtree'] = subtree_r
                self.subtree_count += 1
                self.expression_depth += 1
                self.stack_depth -= 1

            op_idx = action_offset - OFFSET_OP
            if op_idx not in self.used_operators:
                self.used_operators.add(op_idx)
                reward += self.diversity_reward
                components['diversity'] = self.diversity_reward

        if self.expression_depth > 0:
            complexity_r = self.complexity_reward * min(self.expression_depth, 10) / 10
            reward += complexity_r
            components['complexity'] = complexity_r

        return reward, {
            'type': 'dense',
            'components': components,
            'stack_depth': self.stack_depth,
            'expression_depth': self.expression_depth,
            'subtree_count': self.subtree_count
        }


class AlphaEnvWrapper(gym.Wrapper):
    state: np.ndarray
    env: AlphaEnvCore
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Box
    counter: int

    def __init__(self,
                 env: AlphaEnvCore,
                 use_dense_reward: bool = True,
                 subtree_reward: float = 0.01,
                 complexity_reward: float = 0.005,
                 syntax_reward: float = 0.002,
                 diversity_reward: float = 0.003,
                 dense_reward_scale: float = 1.0):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(SIZE_ACTION)
        self.observation_space = gym.spaces.Box(
            low=0, high=SIZE_ALL - 1,
            shape=(MAX_EXPR_LENGTH,),
            dtype=np.uint8
        )

        self.use_dense_reward = use_dense_reward
        self.dense_reward_scale = dense_reward_scale

        self.dense_reward_calculator = DenseRewardCalculator(
            subtree_reward=subtree_reward,
            complexity_reward=complexity_reward,
            syntax_reward=syntax_reward,
            diversity_reward=diversity_reward,
        )

        self.episode_dense_reward = 0.0
        self.episode_sparse_reward = 0.0

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self.counter = 0
        self.state = np.zeros(MAX_EXPR_LENGTH, dtype=np.uint8)
        self.env.reset()
        self.dense_reward_calculator.reset()
        self.episode_dense_reward = 0.0
        self.episode_sparse_reward = 0.0
        return self.state, {}

    def step(self, action: int):
        _, sparse_reward, done, truncated, info = self.env.step(self.action(action))

        dense_reward = 0.0
        if self.use_dense_reward:
            is_valid = not done or sparse_reward >= 0
            dense_reward, dense_info = self.dense_reward_calculator.calculate_reward(action, is_valid)
            dense_reward *= self.dense_reward_scale
            info['dense_info'] = dense_info

        if not done:
            self.state[self.counter] = action
            self.counter += 1

        total_reward = self.reward(sparse_reward) + dense_reward
        # print(f"[reward] step={self.counter} action={action} "
        #       f"sparse={sparse_reward:.6f} dense={dense_reward:.6f} total={total_reward:.6f} "
        #       f"dense_info={info.get('dense_info')}")

        self.episode_dense_reward += dense_reward
        self.episode_sparse_reward += sparse_reward

        info['dense_reward'] = dense_reward
        info['sparse_reward'] = sparse_reward

        if done:
            info['episode_dense_reward'] = self.episode_dense_reward
            info['episode_sparse_reward'] = self.episode_sparse_reward
            info['expression_depth'] = self.dense_reward_calculator.expression_depth
            info['subtree_count'] = self.dense_reward_calculator.subtree_count

        return self.state, total_reward, done, truncated, info

    def action(self, action: int) -> Token:
        return action2token(action)

    def reward(self, reward: float) -> float:
        return reward + REWARD_PER_STEP

    def action_masks(self) -> np.ndarray:
        res = np.zeros(SIZE_ACTION, dtype=bool)
        valid = self.env.valid_action_types()
        for i in range(OFFSET_OP, OFFSET_OP + SIZE_OP):
            if valid['op'][OPERATORS[i - OFFSET_OP].category_type()]:
                res[i - 1] = True
        if valid['select'][1]:
            for i in range(OFFSET_FEATURE, OFFSET_FEATURE + SIZE_FEATURE):
                res[i - 1] = True
        if valid['select'][2]:
            for i in range(OFFSET_CONSTANT, OFFSET_CONSTANT + SIZE_CONSTANT):
                res[i - 1] = True
        if valid['select'][3]:
            for i in range(OFFSET_DELTA_TIME, OFFSET_DELTA_TIME + SIZE_DELTA_TIME):
                res[i - 1] = True
        if valid['select'][4]:
            res[OFFSET_SEP - 1] = True
        return res


def AlphaEnv(pool: AlphaPoolBase,
             use_dense_reward: bool = True,
             subtree_reward: float = 0.01,
             complexity_reward: float = 0.005,
             syntax_reward: float = 0.002,
             diversity_reward: float = 0.003,
             dense_reward_scale: float = 1.0,
             **kwargs):
    return AlphaEnvWrapper(
        AlphaEnvCore(pool=pool, **kwargs),
        use_dense_reward=use_dense_reward,
        subtree_reward=subtree_reward,
        complexity_reward=complexity_reward,
        syntax_reward=syntax_reward,
        diversity_reward=diversity_reward,
        dense_reward_scale=dense_reward_scale,
    )

