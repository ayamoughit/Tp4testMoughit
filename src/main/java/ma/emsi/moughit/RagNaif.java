package ma.emsi.moughit;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RagNaif {

    public static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {
        configureLogger();

        Path documentPath = toPath("rag.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(documentPath);

        // Découpage du document en morceaux
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        // Création du modèle d’embedding
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        Response<List<Embedding>> embeddingsResponse = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = embeddingsResponse.content();

        // Stockage des embeddings en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // 💬 Phase 2 : Création de l’assistant (RAG)
        String llmKey = System.getenv("GEMINI_KEY");
        if (llmKey == null || llmKey.isEmpty()) {
            System.err.println("Erreur : la clé GEMINI_KEY n’est pas définie dans les variables d’environnement.");
            System.exit(1);
        }

        ChatModel chatModel = GoogleAiGeminiChatModel
                .builder()
                .apiKey(llmKey)
                .temperature(0.3)
                .modelName("gemini-2.5-flash")
                .logRequestsAndResponses(true)
                .build();

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .contentRetriever(contentRetriever)
                .chatMemory(chatMemory)
                .build();

        System.out.println("Assistant RAG prêt ! Posez vos questions (ou tapez 'exit' pour quitter).");
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("Vous : ");
            String userMessage = scanner.nextLine();

            if ("exit".equalsIgnoreCase(userMessage)) break;

            String assistantResponse = assistant.chat(userMessage);
            System.out.println("Assistant : " + assistantResponse);
        }
        scanner.close();
    }

    private static Path toPath(String resourceName) {
        try {
            URL resourceUrl = RagNaif.class.getClassLoader().getResource(resourceName);
            if (resourceUrl == null) {
                throw new RuntimeException("Ressource introuvable : " + resourceName);
            }
            return Paths.get(resourceUrl.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }
}
