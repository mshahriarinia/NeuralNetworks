function s = indexed_fourgram_to_string( fourgram, vocab )
% This function will return the string representation of fourgrams for
% display and deubg purposes.
% indexed_fourgram_to_string(data.trainData(:, 3), data.vocab)  displays
% the string equivalent for the third train fourgram.

s = [vocab(fourgram(1)) vocab(fourgram(2)) vocab(fourgram(3)) vocab(fourgram(4))];

end

