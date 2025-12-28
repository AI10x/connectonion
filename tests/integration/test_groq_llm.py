
import os
import pytest
from unittest.mock import MagicMock, patch
from connectonion.core.llm import GroqLLM, create_llm

class TestGroqLLM:
    
    @pytest.fixture
    def mock_env(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key-123"}):
            yield

    def test_init_raises_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                GroqLLM()

    def test_init_with_env_key(self, mock_env):
        llm = GroqLLM()
        assert llm.api_key == "test-key-123"
        assert str(llm.client.base_url) == "https://api.groq.com/openai/v1/"

    def test_init_with_arg_key(self):
        llm = GroqLLM(api_key="manual-key")
        assert llm.api_key == "manual-key"

    def test_factory_create(self, mock_env):
        llm = create_llm("gpt-oss 120B")
        assert isinstance(llm, GroqLLM)
        assert llm.model == "gpt-oss 120B"
        
        llm2 = create_llm("llama3-70b-8192")
        assert isinstance(llm2, GroqLLM)

    @patch("openai.OpenAI")
    def test_complete_call(self, mock_openai, mock_env):
        # Setup mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Groq response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        
        mock_client.chat.completions.create.return_value = mock_response
        
        llm = GroqLLM()
        response = llm.complete([{"role": "user", "content": "Hello"}])
        
        assert response.content == "Groq response"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-oss 120B"
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    @patch("openai.OpenAI")
    def test_complete_with_tools(self, mock_openai, mock_env):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_response
        
        llm = GroqLLM()
        tools = [{"name": "test_tool", "description": "test", "parameters": {}}]
        llm.complete([{"role": "user", "content": "Hello"}], tools=tools)
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"

