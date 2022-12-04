from aocd.models import Puzzle
from queue import PriorityQueue


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


if __name__ == '__main__':
    year = 2022
    day = 4
    part = 2
    puzzle = Puzzle(year=year, day=day, )
    fname = "puzzle_" + str(year) + "_" + str(day) + "_" + str(part)
    answer = globals()[fname](puzzle.input_data)
    print(answer)
    if part == 1:
        puzzle.answer_a = answer
    elif part == 2:
        puzzle.answer_b = answer
