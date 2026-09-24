# Some chunks size are greater than the default set size i.e 800
- **Reason:** Bcoz the default separator used by `CharacterTextSplitter()` class is `"\n\n"`. 
- **Solution:** We will use a separator that can let us generate chunks such that their size never exceeds the 800 marks. So we will use `"."` as a separator and i don't think there will be more than 800 tokens in one sentance.
