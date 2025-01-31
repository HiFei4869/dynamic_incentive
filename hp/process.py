# Load the content from celebrity-related questions file
celebrity_questions = [
    "Which actor is known for his roles in 'Pirates of the Caribbean' and 'Edward Scissorhands'?",
    "Who starred as Wolverine in the X-Men film series?",
    "Which singer is often referred to as 'The Queen of Pop'?",
    "Who played the role of Tony Stark in the Marvel Cinematic Universe?",
    "Which actress won an Academy Award for her role in 'La La Land'?",
    "Who is the youngest person to win the Grammy Award for Album of the Year?",
    "Which actor is known for his role as John Wick in the movie franchise?",
    "Who portrayed the role of Katniss Everdeen in 'The Hunger Games' series?",
    "Which actor played Batman in 'The Dark Knight' trilogy?",
    "Who is known for his performances in movies such as 'Titanic' and 'Inception'?",
    "Which actress starred in the TV show 'Friends' as Rachel Green?",
    "Who was the first African American woman to win an Academy Award for Best Actress?",
    "Which actor voiced the character of Woody in the 'Toy Story' franchise?",
    "Who is the lead singer of the band Coldplay?",
    "Who directed the film 'Inception' and the 'Dark Knight' trilogy?",
    "Which actress played Hermione Granger in the Harry Potter film series?",
    "Who is the highest-paid actor in Hollywood, known for his role in 'Fast and Furious'?",
    "Which actor starred in 'The Revenant' and won an Academy Award for his performance?",
    "Who is the author of the 'A Song of Ice and Fire' book series that inspired 'Game of Thrones'?",
    "Which actor is known for playing Captain Jack Sparrow in the 'Pirates of the Caribbean' movies?",
    "Who starred as James Bond in the 2006 movie 'Casino Royale'?",
    "Who is known as 'The King of Pop' and performed hit songs like 'Thriller'?",
    "Which actress won an Oscar for her role in 'Black Swan'?",
    "Who portrayed Jack Dawson in the 1997 film 'Titanic'?",
    "Which singer's real name is Stefani Joanne Angelina Germanotta?",
    "Who played the lead role in the 'Mission Impossible' film series?",
    "Which actor is known for his role in 'Fight Club' and 'Ocean's Eleven'?",
    "Who won an Academy Award for her role in 'Silver Linings Playbook'?",
    "Which famous pop star is married to model Hailey Baldwin?",
    "Who portrayed 'Deadpool' in the Marvel film series?",
    "Which actor starred in 'Mad Max: Fury Road' and 'The Revenant'?",
    "Who played the role of Forrest Gump in the 1994 movie of the same name?",
    "Which actress is known for her roles in 'The Devil Wears Prada' and 'Les Misérables'?",
    "Who is the lead singer of the rock band U2?",
    "Which actor is known for his roles in 'Top Gun' and 'Mission Impossible'?",
    "Who is the creator and writer of the TV series 'Breaking Bad'?",
    "Which singer is known for hits like 'Rolling in the Deep' and 'Someone Like You'?",
    "Which actor portrayed Spider-Man in the 2002 film series directed by Sam Raimi?",
    "Who won an Academy Award for her role in the film 'Room'?",
    "Which actor played Frodo Baggins in the 'Lord of the Rings' trilogy?",
    "Who is the highest-paid female musician in the world?",
    "Which actor starred in both 'The Social Network' and 'Zombieland'?",
    "Who is known for directing 'Pulp Fiction' and 'Kill Bill'?",
    "Which actress starred in the romantic comedy 'Pretty Woman'?",
    "Who played the role of Iron Man in the Marvel Cinematic Universe?",
    "Which actor is famous for his role in 'Die Hard'?",
    "Which pop star had a hit with 'Bad Romance'?",
    "Who directed the 'Jurassic Park' and 'Schindler's List' movies?",
    "Which actress starred in 'Moulin Rouge!' and 'The Others'?"
]
input_file = "./queries.jsonl"
output_file = "./pair_queries.jsonl"
# Reading Harry Potter questions and writing the merged output
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for idx, line in enumerate(infile):
        hp_question = line.strip()
        celebrity_question = celebrity_questions[idx % len(celebrity_questions)]  # Loop over the list if it exceeds
        merged_query = f'{{"query": "{hp_question.rstrip()} {celebrity_question.lower()} "}}'
        outfile.write(merged_query + "}\n")
