from aocd.models import Puzzle
from queue import PriorityQueue
from collections import deque
import math
import numpy as np
from heapq import nlargest
from functools import cmp_to_key
import string


def puzzle_2022_1_1(input=None):
    maxi = 0
    buff = 0
    for line in input.split('\n'):
        if line == "":
            if buff > maxi:
                maxi = buff
            buff = 0
        else:
            buff += int(line)
    if buff > maxi:
        maxi = buff
    return maxi


def puzzle_2022_1_2(input=None):
    q = PriorityQueue(maxsize=4)
    ans = 0
    buff = 0
    for line in input.split('\n'):
        if line == "":
            q.put((buff, buff))
            if (q.full()):
                q.get()
            buff = 0
        else:
            buff += int(line)
    for i in range(3):
        ans += q.get()[0]
    return ans


def puzzle_2022_2_1(input=None):
    score = 0
    for line in input.split('\n'):
        opponent = ord(line[0]) - ord('A')
        me = ord(line[2]) - ord('X')
        score += 1 + me + ((((me - opponent) % 3) + 1) % 3) * 3
    return score


def puzzle_2022_2_2(input=None):
    score = 0
    for line in input.split('\n'):
        opponent = ord(line[0]) - ord('A')
        result = ord(line[2]) - ord('X')
        score += 1 + ((opponent + (result + 2) % 3) % 3) + result * 3
    return score


def puzzle_2022_3_1(input=None):
    def evaluate(a):
        num = ord(a)
        return (1 + num - ord('a')) if num >= ord('a') else (27 + num - ord('A'))

    ans = 0
    for line in input.split('\n'):
        length = len(line) // 2
        first_str = line[:length]
        second_str = line[length:]
        first_dic = {}
        second_dic = {}
        for i in range(length):
            a = first_str[i]
            b = second_str[i]
            if (a not in first_dic):
                first_dic[a] = evaluate(a)
            if (b not in second_dic):
                second_dic[b] = evaluate(b)
        difference_dic = dict(first_dic.items() & second_dic.items())
        for i in difference_dic:
            ans += difference_dic[i]
    return ans


def puzzle_2022_3_2(input=None):
    def evaluate(a):
        num = ord(a)
        return (1 + num - ord('a')) if num >= ord('a') else (27 + num - ord('A'))

    ans = 0
    for line in zip(input.split('\n')[::3], input.split('\n')[1::3], input.split('\n')[2::3]):
        dics = [{}, {}, {}]
        for i in range(3):
            for j in line[i]:
                if (j not in dics[i]):
                    dics[i][j] = evaluate(j)

        difference_dic = dict(dics[0].items() & dics[1].items() & dics[2].items())
        for i in difference_dic:
            ans += difference_dic[i]
    return ans


def puzzle_2022_4_1(input=None):
    ans = 0
    for line in input.split('\n'):
        edges_list = []
        for i in line.split(','):
            for j in i.split('-'):
                edges_list.append(int(j))
        if ((edges_list[2] <= edges_list[0] and edges_list[1] <= edges_list[3]) or (
                edges_list[0] <= edges_list[2] and edges_list[3] <= edges_list[1])):
            ans += 1
    return ans


def puzzle_2022_4_2(input=None):
    ans = 0
    for line in input.split('\n'):
        edges_list = []
        for i in line.split(','):
            for j in i.split('-'):
                edges_list.append(int(j))
        if (edges_list[1] > edges_list[3]):
            (edges_list[0], edges_list[1], edges_list[2], edges_list[3]) = (
                edges_list[2], edges_list[3], edges_list[0], edges_list[1])
        if (edges_list[2] <= edges_list[1]):
            ans += 1
    return ans


def puzzle_2022_5_1(input=None):
    move_flag = False
    nr_of_stacks = (len(input.split('\n')[0]) + 1) // 4
    stacks = [deque() for _ in range(nr_of_stacks)]
    ans = ""
    for line in input.split('\n'):
        if (not move_flag):
            if (len(line) == 0):
                move_flag = True
            else:
                for i in range(nr_of_stacks):
                    if (line[1 + 4 * i] != ' '):
                        stacks[i].appendleft(line[1 + 4 * i])
        else:
            command = line.split(" ")
            for i in range(int(command[1])):
                stacks[int(command[5]) - 1].append(stacks[int(command[3]) - 1].pop())
    for st in stacks:
        ans += st.pop()
    return ans


def puzzle_2022_5_2(input=None):
    move_flag = False
    nr_of_stacks = (len(input.split('\n')[0]) + 1) // 4
    stacks = [[] for _ in range(nr_of_stacks)]
    ans = ""
    for line in input.split('\n'):
        if (not move_flag):
            if (len(line) == 0):
                move_flag = True
            else:
                for i in range(nr_of_stacks):
                    if (line[1 + 4 * i] != ' '):
                        stacks[i].insert(0, line[1 + 4 * i])
        else:
            command = line.split(" ")
            stacks[int(command[5]) - 1] += (stacks[int(command[3]) - 1][-int(command[1]):])
            stacks[int(command[3]) - 1] = stacks[int(command[3]) - 1][:-int(command[1])]
    for st in stacks:
        ans += st.pop()
    return ans


def puzzle_2022_6_1(input=None):
    l = 0
    for r in range(0, len(input) - 1):
        if (r - l >= 4):
            return r
        for i in range(l, r + 1):
            if (input[i] == input[r + 1]):
                l = i + 1
                break


def puzzle_2022_6_2(input=None):
    l = 0
    for r in range(0, len(input) - 1):
        if (r - l >= 14):
            return r
        for i in range(l, r + 1):
            if (input[i] == input[r + 1]):
                l = i + 1
                break


class File:

    def __init__(self, name, type, parent, size=None, children=None):
        self.name = name
        self.type = type
        self.parent = parent
        self.size = size
        if (children == None):
            self.children = []
        else:
            self.children = children

    def get_size(self):
        if (self.size == None):
            self.size = sum([_.get_size() for _ in self.children])
        return self.size

    def size_recursion(self, limit):
        if (self.type == "txt"):
            return 0
        ret_size = 0
        if (self.get_size() <= limit):
            ret_size += self.get_size()
        return ret_size + sum([_.size_recursion(limit) for _ in self.children])

    def smol_recursion(self, limit):
        if (self.type == "txt"):
            return math.inf
        child_value = min([_.smol_recursion(limit) for _ in self.children])
        child_value = math.inf if child_value == 0 else child_value
        if (self.get_size() <= limit):
            return child_value
        else:
            return min(self.get_size(), child_value)

    def get_child(self, child_name):
        for i in self.children:
            if (i.name == child_name):
                return i
        print("no child ", child_name, " found")
        return None

    def print_tree(self, offset=0):
        if (offset >= 3):
            return
        print(" " * offset, "- ", self.name, " ( ", self.type, " )")
        if (self.type == "dir"):
            for i in self.children:
                i.print_tree(offset + 2)


def puzzle_2022_7_1(input=None):
    root = File("root", "dir", None)
    wd = None
    for line in input.split('\n'):
        com = line.split(' ')
        if (com[1] == "cd"):
            if (com[2] == "/"):
                wd = root
            elif (com[2] == ".."):
                wd = wd.parent
            else:
                wd = wd.get_child(com[2])
        elif (com[0] != "$"):
            if (com[0] == "dir"):
                wd.children.append(File(com[1], "dir", wd))
            else:
                wd.children.append(File(com[1], "txt", wd, int(com[0])))

    return root.size_recursion(100000)


def puzzle_2022_7_2(input=None):
    root = File("root", "dir", None)
    wd = None
    for line in input.split('\n'):
        com = line.split(' ')
        if (com[1] == "cd"):
            if (com[2] == "/"):
                wd = root
            elif (com[2] == ".."):
                wd = wd.parent
            else:
                wd = wd.get_child(com[2])
        elif (com[0] != "$"):
            if (com[0] == "dir"):
                wd.children.append(File(com[1], "dir", wd))
            else:
                wd.children.append(File(com[1], "txt", wd, int(com[0])))
    print(30000000 - (70000000 - root.get_size()))
    return root.smol_recursion(30000000 - (70000000 - root.get_size()))


def puzzle_2022_8_1(input=None):
    input_array = np.array(list(map(lambda x: list(map(int, [*x])), input.split('\n'))))
    length = input_array.shape[0]
    visibility_array = np.zeros(shape=(length, length, 4))
    for i in [0, -1]:
        visibility_array[i, :, :] = True
        visibility_array[:, i, :] = True
    for i in range(1, length):
        visibility_array[i, 1:-1, 0] = input_array[i, 1:-1] > np.max(input_array[:i, 1:-1], axis=0)
        visibility_array[1:-1, i, 2] = input_array[1:-1, i] > np.max(input_array[1:-1, :i], axis=1)
        visibility_array[length - i - 1, 1:-1, 1] = input_array[length - i - 1, 1:-1] > np.max(
            input_array[(length - i):, 1:-1], axis=0)
        visibility_array[1:-1, length - i - 1, 3] = input_array[1:-1, length - i - 1] > np.max(
            input_array[1:-1, (length - i):], axis=1)
    return np.sum(np.any(visibility_array, axis=2))


def puzzle_2022_8_2(input=None):
    input_array = np.array(list(map(lambda x: list(map(int, [*x])), input.split('\n'))))
    length = input_array.shape[0]
    visibility_array = np.zeros(shape=(length, length, 4), dtype=int)
    last_seen = np.zeros(shape=(10, 4))
    for i in range(0, length):
        last_seen = np.zeros(shape=(10, 4))
        last_seen[:, [1, 3]] = length - 1
        for j in range(0, length):
            visibility_array[i, j, 0] = j - max(last_seen[input_array[i, j]:, 0])
            last_seen[input_array[i, j], 0] = j
            visibility_array[j, i, 2] = j - max(last_seen[input_array[j, i]:, 2])
            last_seen[input_array[j, i], 2] = j
            visibility_array[i, length - j - 1, 1] = min(last_seen[input_array[i, length - j - 1]:, 1]) - (
                    length - j - 1)
            last_seen[input_array[i, (length - j - 1)], 1] = (length - j - 1)
            visibility_array[length - j - 1, i, 3] = min(last_seen[input_array[length - j - 1, i]:, 3]) - (
                    length - j - 1)
            last_seen[input_array[(length - j - 1), i], 3] = (length - j - 1)
    return np.max(np.prod(visibility_array, axis=2))


def puzzle_2022_9_1(input=None):
    Head = np.array([0, 0])
    Tail = np.array([0, 0])
    Dict_positions = {(0, 0): True}
    for line in input.split('\n'):
        word = line.split(' ')
        if (word[0] == 'D'):
            change_vector = [0, -1]
        elif (word[0] == 'U'):
            change_vector = [0, 1]
        elif (word[0] == 'L'):
            change_vector = [-1, 0]
        elif (word[0] == 'R'):
            change_vector = [1, 0]
        for i in range(int(word[1])):
            Head += change_vector
            if (np.any(np.abs(Head - Tail) >= 2)):
                Tail += np.sign(Head - Tail)
                Dict_positions[tuple(Tail)] = True
    return len(Dict_positions)


def puzzle_2022_9_2(input=None):
    Snake = np.zeros(shape=(10, 2))
    Dict_positions = {(0, 0): True}
    for line in input.split('\n'):
        word = line.split(' ')
        if (word[0] == 'D'):
            change_vector = [0, -1]
        elif (word[0] == 'U'):
            change_vector = [0, 1]
        elif (word[0] == 'L'):
            change_vector = [-1, 0]
        elif (word[0] == 'R'):
            change_vector = [1, 0]
        for i in range(int(word[1])):
            Snake[0] += change_vector
            for i in range(0, 9):
                if (np.any(np.abs(Snake[i] - Snake[i + 1]) >= 2)):
                    Snake[i + 1] += np.sign(Snake[i] - Snake[i + 1])
            Dict_positions[tuple(Snake[9])] = True
    return len(Dict_positions)


def puzzle_2022_10_1(input=None):
    lines = input.split('\n')
    xvalue = np.full(shape=(len(lines) * 2 + 1), fill_value=1)
    current_clock = 1
    for line in lines:
        words = line.split(' ')
        if (words[0] == "noop"):
            current_clock += 1
        else:
            xvalue[current_clock + 2:] += int(words[1])
            current_clock += 2
    xvalue *= np.arange(0, len(xvalue))
    return np.sum(xvalue[[20, 60, 100, 140, 180, 220]])


def puzzle_2022_10_2(input=None):
    lines = input.split('\n')
    xvalue = np.full(shape=(len(lines) * 2 + 1), fill_value=1)
    current_clock = 1
    for line in lines:
        words = line.split(' ')
        if (words[0] == "noop"):
            current_clock += 1
        else:
            xvalue[current_clock + 2:] += int(words[1])
            current_clock += 2
    screen = np.zeros(shape=(6, 40), dtype=int)
    for i in range(6):
        for j in range(1, 41):
            if (j == xvalue[i * 40 + j] or j == xvalue[i * 40 + j] + 1 or j == xvalue[i * 40 + j] + 2):
                screen[i, j - 1] = 1
    return screen


class Monkey:
    def __init__(self, items=None, operation=None, post_operation=None, modulo=1, test_results=None):
        self.items = items
        self.operation = operation
        self.post_operation = post_operation
        self.modulo = modulo
        self.test_results = test_results
        self.monkey_bussiness = 0

    def inspect(self, item):
        for op in [self.operation, self.post_operation]:
            first = str(item) if op[0] == "old" else op[0]
            second = str(item) if op[2] == "old" else op[2]
            item = eval(first + op[1] + second)
        self.monkey_bussiness += 1
        return item

    def test(self, item):
        return self.test_results[0] if item % self.modulo == 0 else self.test_results[1]


def puzzle_2022_11_1(input=None):
    monkeys = []
    for monkey in input.split("\n\n"):
        monkey_data = monkey.split('\n')
        starting_items = list(map(int, monkey_data[1].split(':')[1].split(",")))
        operation = monkey_data[2].split(' ')[-3:]
        modulo = int(monkey_data[3].split(' ')[-1])
        test_results = [int(monkey_data[4].split(' ')[-1]), int(monkey_data[5].split(' ')[-1])]
        post_operation = ["old", "//", "3"]
        monkeys.append(Monkey(starting_items, operation, post_operation, modulo, test_results))
    for round_nr in range(20):
        for monkey in monkeys:
            for item in monkey.items:
                worry_level = monkey.inspect(item)
                monkeys[monkey.test(worry_level)].items.append(worry_level)
            monkey.items = []

    return int(np.product(nlargest(2, [_.monkey_bussiness for _ in monkeys])))


def puzzle_2022_11_2(input=None):
    monkeys = []
    for monkey in input.split("\n\n"):
        monkey_data = monkey.split('\n')
        starting_items = list(map(int, monkey_data[1].split(':')[1].split(",")))
        operation = monkey_data[2].split(' ')[-3:]
        modulo = int(monkey_data[3].split(' ')[-1])
        test_results = [int(monkey_data[4].split(' ')[-1]), int(monkey_data[5].split(' ')[-1])]
        post_operation = ["old", "%"]
        monkeys.append(Monkey(starting_items, operation, post_operation, modulo, test_results))
    macro_modulo = int(np.product([_.modulo for _ in monkeys]))
    for monkey in monkeys:
        monkey.post_operation.append(str(macro_modulo))
    for round_nr in range(10000):
        for monkey in monkeys:
            for item in monkey.items:
                worry_level = monkey.inspect(item)
                monkeys[monkey.test(worry_level)].items.append(worry_level)
            monkey.items = []

    return int(np.product(nlargest(2, [_.monkey_bussiness for _ in monkeys])))


def puzzle_2022_12_1(input=None):
    input_array = np.array(list(map(lambda x: list(map(ord, [*x])), input.split('\n'))))
    directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    start_point = np.where(input_array == ord('S'))
    start_point = (start_point[0][0], start_point[1][0])
    input_array[start_point] = ord('a')

    end_point = np.where(input_array == ord('E'))
    end_point = (end_point[0][0], end_point[1][0])
    input_array[end_point] = max(input_array[(directions + end_point).T[0], (directions + end_point).T[1]])

    g_score = np.full_like(input_array, input_array.shape[0] * input_array.shape[1])
    g_score[start_point] = 0
    f_score = np.full_like(input_array, input_array.shape[0] * input_array.shape[1])
    f_score[start_point] = np.maximum(np.sum(abs(np.array(end_point) - start_point)),
                                      input_array[end_point] - input_array[start_point])
    points_to_discover = PriorityQueue()
    points_to_discover.put((f_score[start_point], start_point))
    while not points_to_discover.empty():
        current = points_to_discover.get()[1]
        if current == end_point:
            return g_score[end_point]
        for direction in directions:
            neighbor = tuple(current + direction)
            if (neighbor[0] >= 0 and neighbor[0] < input_array.shape[0] and neighbor[1] >= 0 and neighbor[1] <
                    input_array.shape[1]):
                if (input_array[neighbor] - input_array[current] <= 1):
                    if (g_score[current] + 1 < g_score[neighbor]):
                        g_score[neighbor] = g_score[current] + 1
                        f_score[neighbor] = g_score[current] + 1 + np.maximum(
                            np.sum(abs(np.array(end_point) - neighbor)), input_array[end_point] - input_array[neighbor])
                        points_to_discover.put((f_score[neighbor], neighbor))


def puzzle_2022_12_2(input=None):
    input_array = np.array(list(map(lambda x: list(map(ord, [*x])), input.split('\n'))))
    directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    start_point = np.where(input_array == ord('S'))
    start_point = (start_point[0][0], start_point[1][0])
    input_array[start_point] = ord('a')

    end_point = np.where(input_array == ord('E'))
    end_point = (end_point[0][0], end_point[1][0])
    input_array[end_point] = max(input_array[(directions + end_point).T[0], (directions + end_point).T[1]])

    g_score = np.full_like(input_array, input_array.shape[0] * input_array.shape[1])
    g_score[end_point] = 0

    points_to_discover = PriorityQueue()
    points_to_discover.put((g_score[end_point], end_point))
    while not points_to_discover.empty():
        current = points_to_discover.get()[1]
        if input_array[current] == ord('a'):
            return g_score[current]
        for direction in directions:
            neighbor = tuple(current + direction)
            if (neighbor[0] >= 0 and neighbor[0] < input_array.shape[0] and neighbor[1] >= 0 and neighbor[1] <
                    input_array.shape[1]):
                if (input_array[current] - input_array[neighbor] <= 1):
                    if (g_score[current] + 1 < g_score[neighbor]):
                        g_score[neighbor] = g_score[current] + 1
                        points_to_discover.put((g_score[neighbor], neighbor))


def puzzle_2022_13_1(input=None):
    def list_parser(str_input):
        stack = [[]]
        for element in str_input.split(','):
            num_str = ""
            for letter in element:
                if (letter == '['):
                    stack.append([])
                elif (letter == ']'):
                    if (num_str != ""):
                        stack[-1].append(int(num_str))
                        num_str = ""
                    temp = stack.pop()
                    stack[-1].append(temp)
                else:
                    num_str += letter
            if (num_str != ''):
                stack[-1].append(int(num_str))
        return stack[0][0]

    def compare(left, right):
        iflint = type(left) == int
        ifrint = type(right) == int
        if (iflint or ifrint):
            if (iflint and ifrint):
                return np.sign(right - left)
            elif iflint:
                return compare([left], right)
            elif ifrint:
                return compare(left, [right])
        else:
            for i in range(min(len(left), len(right))):
                if (compare(left[i], right[i]) != 0):
                    return compare(left[i], right[i])
            return np.sign(len(right) - len(left))

    sum = 0
    for count, pairs in enumerate(input.split('\n\n'), start=1):
        pairs = pairs.split('\n')
        if (compare(list_parser(pairs[0]), list_parser(pairs[1])) == 1):
            sum += count
    return sum


def puzzle_2022_13_2(input=None):
    def list_parser(str_input):
        stack = [[]]
        for element in str_input.split(','):
            num_str = ""
            for letter in element:
                if (letter == '['):
                    stack.append([])
                elif (letter == ']'):
                    if (num_str != ""):
                        stack[-1].append(int(num_str))
                        num_str = ""
                    temp = stack.pop()
                    stack[-1].append(temp)
                else:
                    num_str += letter
            if (num_str != ''):
                stack[-1].append(int(num_str))
        return stack[0][0]

    def compare(left, right):
        iflint = type(left) == int
        ifrint = type(right) == int
        if (iflint or ifrint):
            if (iflint and ifrint):
                return np.sign(right - left)
            elif iflint:
                return compare([left], right)
            elif ifrint:
                return compare(left, [right])
        else:
            for i in range(min(len(left), len(right))):
                if (compare(left[i], right[i]) != 0):
                    return compare(left[i], right[i])
            return np.sign(len(right) - len(left))

    input_list = list(map(list_parser, input.replace("\n\n", "\n").split('\n')))
    input_list.append([[2]])
    input_list.append([[6]])
    input_list.sort(key=cmp_to_key(compare), reverse=True)
    return (input_list.index([[6]]) + 1) * (input_list.index([[2]]) + 1)


def puzzle_2022_14_1(input=None):
    input_list = list(
        map(lambda x: list(map(lambda y: np.array(list(map(int, y.split(',')))), x.split('->'))), input.split('\n')))
    grid = np.zeros(shape=(1000, 1000), dtype=int)
    for line in input_list:
        for pair_num in range(len(line) - 1):
            pair_diffrence = line[pair_num + 1] - line[pair_num]
            if (pair_diffrence[0] == 0):
                bigger = line[pair_num + 1] if pair_diffrence[1] > 0 else line[pair_num]
                grid[line[pair_num][0], (bigger - abs(pair_diffrence))[1]: bigger[1] + 1] = 1
            elif (pair_diffrence[1] == 0):
                bigger = line[pair_num + 1] if pair_diffrence[0] > 0 else line[pair_num]
                grid[(bigger - abs(pair_diffrence))[0]: bigger[0] + 1, line[pair_num][1]] = 1
    for i in range(1000 * 1000):
        x = 500
        y = 0
        for y in range(999):
            if (grid[x, y + 1] == 1):
                if (grid[x - 1, y + 1] == 1):
                    if (grid[x + 1, y + 1] == 1):
                        break
                    else:
                        x += 1
                else:
                    x -= 1
        if (y == 998):
            return i
        else:
            grid[x, y] = 1


def puzzle_2022_14_2(input=None):
    input_list = list(
        map(lambda x: list(map(lambda y: np.array(list(map(int, y.split(',')))), x.split('->'))), input.split('\n')))
    max_y = max([num for line in input_list for pair in line for num in pair][1::2])
    grid = np.zeros(shape=(1000, max_y + 3), dtype=int)
    grid[:, -1] = 1
    for line in input_list:
        for pair_num in range(len(line) - 1):
            pair_diffrence = line[pair_num + 1] - line[pair_num]
            if (pair_diffrence[0] == 0):
                bigger = line[pair_num + 1] if pair_diffrence[1] > 0 else line[pair_num]
                grid[line[pair_num][0], (bigger - abs(pair_diffrence))[1]: bigger[1] + 1] = 1
            elif (pair_diffrence[1] == 0):
                bigger = line[pair_num + 1] if pair_diffrence[0] > 0 else line[pair_num]
                grid[(bigger - abs(pair_diffrence))[0]: bigger[0] + 1, line[pair_num][1]] = 1

    for i in range(1000 * (max_y + 3)):
        x = 500
        y = 0
        if (grid[x, y] == 1):
            return i
        for y in range(max_y + 2):
            if (grid[x, y + 1] == 1):
                if (grid[x - 1, y + 1] == 1):
                    if (grid[x + 1, y + 1] == 1):
                        break
                    else:
                        x += 1
                else:
                    x -= 1
        grid[x, y] = 1


if __name__ == '__main__':
    year = 2022
    day = 14
    part = 2
    send = True
    puzzle = Puzzle(year=year, day=day, )
    fname = "puzzle_" + str(year) + "_" + str(day) + "_" + str(part)
    answer = globals()[fname](puzzle.input_data)
    print(answer)
    if (send):
        if part == 1:
            puzzle.answer_a = answer
        elif part == 2:
            puzzle.answer_b = answer
