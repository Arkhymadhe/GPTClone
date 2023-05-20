def take_data():
    file_name = input("Please type in file name: ").lower()
    prompts = list()

    prompt = input("Please type in your prompt here: ")

    while prompt:
        prompts.append(prompt)
        prompt = input("Please type in another prompt here: ")

    with open(file_name, "w") as f:
        for prompt in prompts:
            f.write(prompt + ",\n")

    return file_name


def batch_data(file_name):
    batched_data = file_name
    return batched_data


def load_data(file_name):
    with open(file_name, "r") as f:
        lines = list(map(lambda x: x[:-2], f.readlines()))

    return lines


if __name__ == "__main__":
    fname = "text.txt"
    print(load_data(fname))
