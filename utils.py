import csv


def generate_csv():
    facts = [
        {"id": 1, "fact": "The first human to orbit Earth was Yuri Gagarin in 1961."},
        {
            "id": 2,
            "fact": "The Apollo 11 mission landed the first humans on the Moon in 1969.",
        },
        {
            "id": 3,
            "fact": "The Hubble Space Telescope was launched in 1990 and has provided stunning images of the universe.",
        },
        {
            "id": 4,
            "fact": "Mars is the most explored planet in the solar system, with multiple rovers sent by NASA.",
        },
        {
            "id": 5,
            "fact": "The International Space Station (ISS) has been continuously occupied since November 2000.",
        },
        {
            "id": 6,
            "fact": "Voyager 1 is the farthest human-made object from Earth, launched in 1977.",
        },
        {
            "id": 7,
            "fact": "SpaceX, founded by Elon Musk, is the first private company to send humans to orbit.",
        },
        {
            "id": 8,
            "fact": "The James Webb Space Telescope, launched in 2021, is the successor to the Hubble Telescope.",
        },
        {"id": 9, "fact": "The Milky Way galaxy contains over 100 billion stars."},
        {
            "id": 10,
            "fact": "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
        },
    ]

    with open("./rag/space_facts.csv", mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "fact"])
        writer.writeheader()
        writer.writerows(facts)

    print("CSV file 'space_facts.csv' created successfully!")


generate_csv()
