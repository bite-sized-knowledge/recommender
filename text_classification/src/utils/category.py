CATEGORY_DICT = {
    'Frontend': 1,
    'Backend': 2,
    'Mobile Engineering': 3,
    'AI / ML': 4,
    'Database': 5,
    'Security / Network': 6,
    'Design': 7,
    'Product Manager': 8,
    'DevOps / Infra': 9,
    'Hardware / IoT': 10,
    'QA / Test Engineer': 11,
    'Culture': 12,
    'etc' : 13
}

CATEGORY = [
    'Frontend'
    'Backend'
    'Mobile Engineering'
    'AI / ML'
    'Database'
    'Security / Network'
    'Design'
    'Product Manager'
    'DevOps / Infra'
    'Hardware / IoT'
    'QA / Test Engineer'
    'Culture'
    'etc'
]

def category_to_idx(category):
    return CATEGORY.index(category)+1

def idx_to_category(idx):
    return CATEGORY[idx]
