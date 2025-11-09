package ma.emsi.moughit;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

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

    // --- Fonction de configuration du logger ---
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {


        configureLogger();

        // Phase 1 : Ingestion du document
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Path documentPath = toPath("rag.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(documentPath, documentParser);

        // Découper le document
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        List<TextSegment> segments = splitter.split(document);

        // Créer les embeddings
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        // Stocker les embeddings
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // Phase 2 : Retrieval et génération
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Mémoire de chat
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // Modèle Gemini avec logs activés
        ChatModel model = GoogleAiGeminiChatModel
                .builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .modelName("gemini-2.5-flash")
                .build();

        // Assistant LangChain4j
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .contentRetriever(contentRetriever)
                .chatMemory(chatMemory)
                .build();

        // --- Interaction utilisateur ---
        System.out.println("Assistant is ready. Ask your questions.");
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.print("You: ");
            String userMessage = scanner.nextLine();

            if ("exit".equalsIgnoreCase(userMessage)) {
                break;
            }

            String answer = assistant.chat(userMessage);
            System.out.println("Assistant: " + answer);
        }

        scanner.close();
    }

    private static Path toPath(String fileName) {
        try {
            URL fileUrl = RagNaif.class.getClassLoader().getResource(fileName);
            if (fileUrl == null) {
                throw new RuntimeException("File not found: " + fileName);
            }
            return Paths.get(fileUrl.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }
}
