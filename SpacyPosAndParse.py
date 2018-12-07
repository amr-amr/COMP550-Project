import spacy
import pandas as pd
import tqdm


class SpacyPosAndParse:
    def __init__(self, spacy_model="en_core_web_lg"):
        self.pos_dict = self.build_pos_dict()
        self.nlp = spacy.load(spacy_model)

    def build_pos_dict(self):
        pos_dict = {'ADJ': 0,
                    'ADP': 1,
                    'ADV': 2,
                    'AUX': 3,
                    'CONJ': 4,
                    'CCONJ': 4,
                    'DET': 5,
                    'INTJ': 6,
                    'NOUN': 7,
                    'NUM': 8,
                    'PART': 9,
                    'PRON': 10,
                    'PROPN': 11,
                    'PUNCT': 12,
                    'SCONJ': 13,
                    'SYM': 14,
                    'VERB': 15,
                    'X': 16 }
        return pos_dict

    def parse_text(self, text):
        pos_tags = []
        parse_tree = []

        doc = self.nlp(text)
        for token in doc:
            # parse pos tags
            pos_tag = token.pos_
            try: i = self.pos_dict[pos_tag]
            except: i = 16
            pos_tags.append(i)

            # dependency parse
            dep_and_head = (token.dep_, token.head.i)
            parse_tree.append(dep_and_head)

        return pos_tags, parse_tree


if __name__ == '__main__':
    reviews =   ["<START> this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert redford's is an amazing actor and now the same being director norman's father came from the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for retail and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad and you know what they say if you cry at a film it must have been good and this definitely was also congratulations to the two little boy's that played the part's of norman and paul they were just brilliant children are often left out of the praising list i think because the stars that play them all grown up are such a big profile for the whole film but these children are amazing and should be praised for what they have done don't you think the whole story was so lovely because it was true and was someone's life after all that was shared with us all",
                 "<START> big hair big boobs bad music and a giant safety pin these are the words to best describe this terrible movie i love cheesy horror movies and i've seen hundreds but this had got to be on of the worst ever made the plot is paper thin and ridiculous the acting is an abomination the script is completely laughable the best is the end showdown with the cop and how he worked out who the killer is it's just so damn terribly written the clothes are sickening and funny in equal measures the hair is big lots of boobs bounce men wear those cut tee shirts that show off their stomachs sickening that men actually wore them and the music is just synthesiser trash that plays over and over again in almost every scene there is trashy music boobs and paramedics taking away bodies and the gym still doesn't close for bereavement all joking aside this is a truly bad film whose only charm is to look back on the disaster that was the 80's and have a good old laugh at how bad everything was back then",
                 "<START> this has to be one of the worst films of the 1990s when my friends i were watching this film being the target audience it was aimed at we just sat watched the first half an hour with our jaws touching the floor at how bad it really was the rest of the time everyone else in the theatre just started talking to each other leaving or generally crying into their popcorn that they actually paid money they had earnt working to watch this feeble excuse for a film it must have looked like a great idea on paper but on film it looks like no one in the film has a clue what is going on crap acting crap costumes i can't get across how embarrasing this is to watch save yourself an hour a bit of your life",
                 "<START> the scots excel at storytelling the traditional sort many years after the event i can still see in my mind's eye an elderly lady my friend's mother retelling the battle of culloden she makes the characters come alive her passion is that of an eye witness one to the events on the sodden heath a mile or so from where she lives br br of course it happened many years before she was born but you wouldn't guess from the way she tells it the same story is told in bars the length and breadth of scotland as i discussed it with a friend one night in mallaig a local cut in to give his version the discussion continued to closing time br br stories passed down like this become part of our being who doesn't remember the stories our parents told us when we were children they become our invisible world and as we grow older they maybe still serve as inspiration or as an emotional reservoir fact and fiction blend with aspiration role models warning stories archetypes magic and mystery br br my name is aonghas like my grandfather and his grandfather before him our protagonist introduces himself to us and also introduces the story that stretches back through generations it produces stories within stories stories that evoke the impenetrable wonder of scotland its rugged mountains shrouded in mists the stuff of legend yet seach'd is rooted in reality this is what gives it its special charm it has a rough beauty and authenticity tempered with some of the finest gaelic singing you will ever hear br br aonghas angus visits his grandfather in hospital shortly before his death he burns with frustration part of him yearns to be in the twenty first century to hang out in glasgow but he is raised on the western shores among a gaelic speaking community br br yet there is a deeper conflict within him he yearns to know the truth the truth behind his grandfather's ancient stories where does fiction end and he wants to know the truth behind the death of his parents br br he is pulled to make a last fateful journey to the summit of one of scotland's most inaccessible mountains can the truth be told or is it all in stories br br in this story about stories we revisit bloody battles poisoned lovers the folklore of old and the sometimes more treacherous folklore of accepted truth in doing so we each connect with angus as he lives the story of his own life br br seachd the inaccessible pinnacle is probably the most honest unpretentious and genuinely beautiful film of scotland ever made like angus i got slightly annoyed with the pretext of hanging stories on more stories but also like angus i forgave this once i saw the 'bigger picture ' forget the box office pastiche of braveheart and its like you might even forego the justly famous dramatisation of the wicker man to see a film that is true to scotland this one is probably unique if you maybe meditate on it deeply enough you might even re evaluate the power of storytelling and the age old question of whether there are some truths that cannot be told but only experienced",
                 "<START> worst mistake of my life br br i picked this movie up at target for 5 because i figured hey it's sandler i can get some cheap laughs i was wrong completely wrong mid way through the film all three of my friends were asleep and i was still suffering worst plot worst script worst movie i have ever seen i wanted to hit my head up against a wall for an hour then i'd stop and you know why because it felt damn good upon bashing my head in i stuck that damn movie in the microwave and watched it burn and that felt better than anything else i've ever done it took american psycho army of darkness and kill bill just to get over that crap i hate you sandler for actually going through with this and ruining a whole day of my life",
                 "<START> please give this one a miss br br kristy swanson and the rest of the cast rendered terrible performances the show is flat flat flat br br i don't know how michael madison could have allowed this one on his plate he almost seemed to know this wasn't going to work out and his performance was quite lacklustre so all you madison fans give this a miss",
                 "<START> this film requires a lot of patience because it focuses on mood and character development the plot is very simple and many of the scenes take place on the same set in frances austen's the sandy dennis character apartment but the film builds to a disturbing climax br br the characters create an atmosphere rife with sexual tension and psychological trickery it's very interesting that robert altman directed this considering the style and structure of his other films still the trademark altman audio style is evident here and there i think what really makes this film work is the brilliant performance by sandy dennis it's definitely one of her darker characters but she plays it so perfectly and convincingly that it's scary michael burns does a good job as the mute young man regular altman player michael murphy has a small part the solemn moody set fits the content of the story very well in short this movie is a powerful study of loneliness sexual repression and desperation be patient soak up the atmosphere and pay attention to the wonderfully written script br br i praise robert altman this is one of his many films that deals with unconventional fascinating subject matter this film is disturbing but it's sincere and it's sure to elicit a strong emotional response from the viewer if you want to see an unusual film some might even say bizarre this is worth the time br br unfortunately it's very difficult to find in video stores you may have to buy it off the internet",
                 "<START> many animation buffs consider wladyslaw starewicz the great forgotten genius of one special branch of the art puppet animation which he invented almost single handedly and as it happened almost accidentally as a young man starewicz was more interested in entomology than the cinema but his unsuccessful attempt to film two stag beetles fighting led to an unexpected breakthrough in film making when he realized he could simulate movement by manipulating beetle carcasses and photographing them one frame at a time this discovery led to the production of amazingly elaborate classic short the cameraman's revenge which he made in russia in 1912 at a time when motion picture animation of all sorts was in its infancy br br the political tumult of the russian revolution caused starewicz to move to paris where one of his first productions coincidentally was a dark political satire variously known as frogland or the frogs who wanted a king a strain of black comedy can be found in almost all of films but here it is very dark indeed aimed more at grown ups who can appreciate the satirical aspects than children who would most likely find the climax upsetting i'm middle aged and found it pretty upsetting myself and indeed prints of the film intended for english speaking viewers of the 1920s were given title cards filled with puns and quips in order to help soften the sharp sting of the finale br br our tale is set in a swamp the frogland commonwealth where the citizens are unhappy with their government and have called a special session to see what they can do to improve matters they decide to beseech jupiter for a king the crowds are impressively animated in this opening sequence it couldn't have been easy to make so many frog puppets look alive simultaneously while jupiter for his part is depicted as a droll white bearded guy in the clouds who looks like he'd rather be taking a nap when jupiter sends them a tree like god who regards them the frogs decide that this is no improvement and demand a different king irritated jupiter sends them a stork br br delighted with this formidable looking new king who towers above them the frogs welcome him with a delegation of formally dressed dignitaries the mayor steps forward to hand him the key to the commonwealth as newsreel cameras record the event to everyone's horror the stork promptly eats the mayor and then goes on a merry rampage swallowing citizens at random a title card dryly reads news of the king's appetite throughout the kingdom when the now terrified frogs once more beseech jupiter for help he loses his temper and showers their community with lightning bolts the moral of our story delivered by a hapless frog just before he is eaten is let well enough alone br br considering the time period when this startling little film was made and considering the fact that it was made by a russian émigré at the height of that beleaguered country's civil war it would be easy to see this as a parable about those events starewicz may or may not have had russia's turmoil in mind when he made frogland but whatever prompted his choice of material the film stands as a cautionary tale of universal application frogland could be the soviet union italy germany or japan in the 1930s or any country of any era that lets its guard down and is overwhelmed by tyranny it's a fascinating film even a charming one in its macabre way but its message is no joke",
                 "<START> i generally love this type of movie however this time i found myself wanting to kick the screen since i can't do that i will just complain about it this was absolutely idiotic the things that happen with the dead kids are very cool but the alive people are absolute idiots i am a grown man pretty big and i can defend myself well however i would not do half the stuff the little girl does in this movie also the mother in this movie is reckless with her children to the point of neglect i wish i wasn't so angry about her and her actions because i would have otherwise enjoyed the flick what a number she was take my advise and fast forward through everything you see her do until the end also is anyone else getting sick of watching movies that are filmed so dark anymore one can hardly see what is being filmed as an audience we are impossibly involved with the actions on the screen so then why the hell can't we have night vision",
                 "<START> like some other people wrote i'm a die hard mario fan and i loved this game br br this game starts slightly boring but trust me it's worth it as soon as you start your hooked the levels are fun and exiting they will hook you 'till your mind turns to mush i'm not kidding this game is also orchestrated and is beautifully done br br to keep this spoiler free i have to keep my mouth shut about details but please try this game it'll be worth it br br story 9 9 action 10 1 it's that good hardness 10 attention grabber 10 average 10"]

    df = pd.DataFrame(columns=['text', 'pos', 'parse'])
    df['text'] = reviews

    sda = SpacyPosAndParse(spacy_model="en_core_web_sm")
    df[['pos', 'parse']] = df.apply(lambda row: sda.parse_text(row['text']), axis=1, result_type='expand')
    print(df)
