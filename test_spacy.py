import spacy

def test_spacy():
    try:
        print("Loading en_core_web_sm model...")
        nlp = spacy.load("en_core_web_sm")
        print("Model loaded successfully!")
        
        text = "Apple is looking at buying U.K. startup for $1 billion"
        doc = nlp(text)
        
        print("\nEntities found:")
        for ent in doc.ents:
            print(f"Entity: {ent.text}, Label: {ent.label_}")
            
        print("\nTokens and POS tags:")
        for token in doc:
            print(f"{token.text:12} {token.pos_:10} {token.dep_}")
            
        print("\nTest passed successfully!")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_spacy()
