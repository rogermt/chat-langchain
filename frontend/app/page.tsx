"use client";

import { ToastContainer } from "react-toastify";
import { ChakraProvider } from "@chakra-ui/react";
import { QueryClient, QueryClientProvider } from "react-query";
import { Client } from "@langchain/langgraph-sdk";

import { ChatWindow } from "./components/ChatWindow";
import { LangGraphClientContext } from "./hooks/useLangGraphClient";
import { API_BASE_URL, LANGCHAIN_API_KEY } from "./utils/constants";

export default function Home() {
  const queryClient = new QueryClient();

  const langGraphClient = new Client({
    apiUrl: API_BASE_URL,
    defaultHeaders: { 
      "x-api-key": LANGCHAIN_API_KEY,
      // Add CORS-related headers
      "Access-Control-Allow-Origin": "*", // Or specify your frontend URL
      "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, x-api-key",
    },
    // Add custom fetch function to handle CORS preflight
    fetch: async (url, options) => {
      // For preflight requests
      if (options.method === 'OPTIONS') {
        return new Response(null, {
          headers: {
            "Access-Control-Allow-Origin": "*", // Or specify your frontend URL
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, x-api-key",
          }
        });
      }
      // For actual requests
      return fetch(url, {
        ...options,
        mode: 'cors', // Explicitly set CORS mode
      });
    },
  });

  return (
    <LangGraphClientContext.Provider value={langGraphClient}>
      <QueryClientProvider client={queryClient}>
        <ChakraProvider>
          <ToastContainer />
          <ChatWindow />
        </ChakraProvider>
      </QueryClientProvider>
    </LangGraphClientContext.Provider>
  );
}