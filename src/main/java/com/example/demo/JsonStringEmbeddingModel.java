package com.example.demo;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.ai.document.Document;
import org.springframework.ai.document.MetadataMode;
import org.springframework.ai.embedding.AbstractEmbeddingModel;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.util.Assert;

import java.util.List;
import java.util.stream.IntStream;

public class JsonStringEmbeddingModel extends AbstractEmbeddingModel {

    private final ObjectMapper mapper = new ObjectMapper();
    private final TypeReference<List<Double>> embeddingType =   new TypeReference<List<Double>>() {};

    @Override
    public EmbeddingResponse call(EmbeddingRequest request) {
        List<String> instructions = request.getInstructions();
        List<Embedding> insEmbeddings =
            IntStream.range(0, instructions.size())
                .mapToObj(idx -> {
                    try {
                        List<Double> ins = mapper.readValue(instructions.get(idx), embeddingType);
                        return new Embedding(ins, idx);
                    } catch (JsonProcessingException e) {
                        throw new RuntimeException(e);
                    }
                }).toList();
        return new EmbeddingResponse(insEmbeddings);
    }

    @Override
    public List<Double> embed(Document document) {
        Assert.notNull(document, "Document must not be null");
        return this.embed(document.getFormattedContent(MetadataMode.NONE));
    }

    @Override
    public int dimensions() {
        return 3;
    }
}
