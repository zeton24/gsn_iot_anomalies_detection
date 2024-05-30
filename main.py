def is_in_list(element, li: list) -> bool:
    return element in li  # todo, nie wolno


def calc_support(itemset, data: list, n_transactions: int) -> float:
    if isinstance(itemset, list):
        temp_support = 0
    elif isinstance(itemset, int):
        temp_support = 0
        for transaction in data:
            if is_in_list(itemset, transaction):
                temp_support += 1
        temp_support /= n_transactions
    else:
        temp_support = None
    return temp_support


def main():
    # Constraints. todo
    # min_sup = input("Please input minimal relative support:\n")
    min_sup = 0.5

    # Path to input file.
    # filepath = input("Please input full path to data file:\n")
    filepath = 'C:\\Users\\amand\\Desktop\\Studia\\4. sem\\Data Mining\\Projekt\\data.csv'

    products = []
    transactions = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            data_line = line.strip().split(",")
            transaction = []
    
            for element in data_line:
                el = element.strip()
                if not is_in_list(el, products):
                    products.append(el)
                transaction.append(products.index(el))  # todo, nie wolno mi tak
            transactions.append(sorted(transaction))

    n_products = len(products)
    n_transactions = len(transactions)

    supports = itemsets = [[]]
    itemsets = [[], []]
    for idx in range(n_products):
        supports[0].append(calc_support(idx, transactions, n_transactions))
        if supports[0][idx] > min_sup:
            itemsets[0].append((idx, supports[0][idx]))

    while itemsets[-1]:
        itemsets.append([])
        # create itemsets
        # calculate supports
        # append frequent ones

    print(itemsets)

    """
    n_rules = 0
    if n_rules:
        print_rules = input(f"{n_rules} rules have been found. Print rules (Y/N)?")
        if print_rules == "Y" or print_rules == "y":
            pass  # todo
        # todo, zapisać
        print("Rules have been saved to file: [todo]")
    else:
        print("No rules have been found.")
    """


main()
