function trimTrailingSlash(value: string) {
  return value.replace(/\/+$/, "");
}

export const appConfig = {
  apiBaseUrl: trimTrailingSlash(import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"),
  appTitle: import.meta.env.VITE_APP_TITLE ?? "RAG Workspace",
  useMockApi: import.meta.env.VITE_USE_MOCK_API === "true",
};
