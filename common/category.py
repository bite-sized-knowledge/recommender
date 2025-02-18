CATEGORY = [
    'web',
    'mobile(android, ios) engineering',
    'hardware & iot',
    'ai & ml & data',
    'security & network',
    'db',
    'devops & infra',
    'game',
    'product manager',
    'design',
    'etc',
    'n/a'
]

def category_to_idx(category):
    return CATEGORY.index(category)

def idx_to_category(idx):
    return CATEGORY[idx]
