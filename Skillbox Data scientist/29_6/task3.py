from collections import deque


# Функция bfs возвращает True, если
# результат поиска в ширину элемента
# desired в дереве tree, начиная с
# элемента start, удачен, и False,
# если нет.
def bfs(tree, start, desired):
    visited = set()
    queue = deque()
    queue.appendleft(start)
    visited.add(start)

    while queue:
        current = queue.pop()
        if current == desired:
            return True
        for neighbour in tree[current]:
            if neighbour not in visited:
                queue.appendleft(neighbour)
                visited.add(neighbour)

    return False


def test_bfs():
    tree = {
        1: [2, 3, 4],
        2: [5, 6],
        3: [],
        4: [7, 8],
        5: [9, 10],
        6: [],
        7: [11, 12],
        8: [],
        9: [],
        10: [],
        11: [],
        12: []
    }
    if bfs(tree, 1, 11):
        print('OK')
    if bfs(tree, 1, 6):
        print('OK')
    if bfs(tree, 2, 10):
        print('OK')
    if not bfs(tree, 2, 11):
        print('OK')
    if bfs(tree, 4, 12):
        print('OK')
    if not bfs(tree, 4, 10):
        print('OK')


def main():
    test_bfs()


if __name__ == '__main__':
    main()