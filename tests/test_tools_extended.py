from __future__ import annotations
import json
import warnings
try:
    from starlette.exceptions import StarletteDeprecationWarning
    warnings.filterwarnings('ignore', category=StarletteDeprecationWarning)
except ImportError:
    warnings.filterwarnings('ignore', category=DeprecationWarning)
import pytest
from unittest.mock import patch, MagicMock
import httpx

class TestWebSearchExtended:

    def test_web_search_clamps_to_10(self):
        from app.tools.web_search import web_search
        with patch('app.tools.web_search.os.getenv', return_value=None), patch('app.tools.web_search._duckduckgo', return_value=[{'title': 't', 'url': 'u', 'snippet': 's'}]) as mock:
            web_search('q', max_results=999)
            assert mock.call_args[0][1] == 10

    def test_web_search_clamps_to_1(self):
        from app.tools.web_search import web_search
        with patch('app.tools.web_search.os.getenv', return_value=None), patch('app.tools.web_search._duckduckgo', return_value=[{'title': 't', 'url': 'u', 'snippet': 's'}]) as mock:
            web_search('q', max_results=-5)
            assert mock.call_args[0][1] == 1
            web_search('q', max_results=0)
            assert mock.call_args[0][1] == 1

    def test_web_search_string_max_results(self):
        from app.tools.web_search import web_search
        with patch('app.tools.web_search.os.getenv', return_value=None), patch('app.tools.web_search._duckduckgo', return_value=[{'title': 't', 'url': 'u', 'snippet': 's'}]) as mock:
            web_search('q', max_results='7')
            assert mock.call_args[0][1] == 7

    def test_web_search_tavily_vs_duckduckgo_selection(self):
        from app.tools import web_search as ws_mod
        with patch('app.tools.web_search.os.getenv') as mock_env, patch('app.tools.web_search._tavily', return_value=[{'title': 't', 'url': 'u', 'snippet': 's'}]) as mock_t, patch('app.tools.web_search._duckduckgo') as mock_d:
            mock_env.return_value = 'key'
            ws_mod.web_search('q')
            mock_t.assert_called_once()
            mock_d.assert_not_called()
            mock_t.reset_mock()
            mock_d.reset_mock()
            mock_env.return_value = None
            mock_d.return_value = [{'title': 't', 'url': 'u', 'snippet': 's'}]
            ws_mod.web_search('q')
            mock_d.assert_called_once()
            mock_t.assert_not_called()

    def test_web_search_exception_contains_type(self):
        from app.tools.web_search import web_search
        with patch('app.tools.web_search.os.getenv', return_value=None), patch('app.tools.web_search._duckduckgo', side_effect=RuntimeError('net down')):
            result = web_search('q')
            data = json.loads(result)
            assert data['error'].startswith('web_search failed: RuntimeError')

    def test_web_search_empty_results_error(self):
        from app.tools.web_search import web_search
        with patch('app.tools.web_search.os.getenv', return_value=None), patch('app.tools.web_search._duckduckgo', return_value=[]):
            result = web_search('q')
            assert 'no results' in json.loads(result)['error']

    def test_web_search_none_results_error(self):
        from app.tools.web_search import web_search
        with patch('app.tools.web_search.os.getenv', return_value=None), patch('app.tools.web_search._duckduckgo', return_value=None):
            result = web_search('q')
            assert 'error' in json.loads(result)

    def test_web_search_truncation(self):
        from app.tools.web_search import web_search, MAX_CHARS
        big = [{'title': 't' * 5000, 'url': 'https://example.com/very/long/url/' + 'a' * 1000, 'snippet': 's' * 5000}] * 10
        with patch('app.tools.web_search.os.getenv', return_value=None), patch('app.tools.web_search._duckduckgo', return_value=big):
            result = web_search('q')
            assert len(result) <= MAX_CHARS

    def test_web_search_result_valid_json(self):
        from app.tools.web_search import web_search
        results = [{'title': 'Hello', 'url': 'https://example.com', 'snippet': 'world'}]
        with patch('app.tools.web_search.os.getenv', return_value=None), patch('app.tools.web_search._duckduckgo', return_value=results):
            result = web_search('hello')
            parsed = json.loads(result)
            assert parsed[0]['title'] == 'Hello'

    def test_duckduckgo_empty_html(self):
        from app.tools.web_search import _duckduckgo
        mock_resp = MagicMock()
        mock_resp.text = '<html><body></body></html>'
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.web_search.httpx.get', return_value=mock_resp):
            results = _duckduckgo('q', 5)
        assert results == []

    def test_duckduckgo_missing_link(self):
        from app.tools.web_search import _duckduckgo
        html = '<div class="result"><span>no link</span><a class="result__snippet">snip</a></div>'
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.web_search.httpx.get', return_value=mock_resp):
            results = _duckduckgo('q', 5)
        assert results == []

    def test_duckduckgo_missing_href(self):
        from app.tools.web_search import _duckduckgo
        html = '<div class="result"><a class="result__a">title</a></div>'
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.web_search.httpx.get', return_value=mock_resp):
            results = _duckduckgo('q', 5)
        assert results == []

    def test_duckduckgo_no_snippet(self):
        from app.tools.web_search import _duckduckgo
        html = '<div class="result"><a class="result__a" href="https://example.com">Title</a></div>'
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.web_search.httpx.get', return_value=mock_resp):
            results = _duckduckgo('q', 5)
        assert results[0]['snippet'] == ''
        assert results[0]['title'] == 'Title'

    def test_duckduckgo_respects_max_results(self):
        from app.tools.web_search import _duckduckgo
        html = ''.join([f'<div class="result"><a class="result__a" href="https://example.com/{i}">Title {i}</a><a class="result__snippet">snip {i}</a></div>' for i in range(10)])
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.web_search.httpx.get', return_value=mock_resp):
            results = _duckduckgo('q', 3)
        assert len(results) == 3

    def test_duckduckgo_uddg_decode(self):
        from app.tools.web_search import _duckduckgo
        html = '<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ftarget.com%2Fpath%3Fq%3D1&rut=abc">Title</a><a class="result__snippet">snip</a></div>'
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.web_search.httpx.get', return_value=mock_resp):
            results = _duckduckgo('q', 5)
        assert results[0]['url'] == 'https://target.com/path?q=1'

    def test_duckduckgo_div_snippet_variant(self):
        from app.tools.web_search import _duckduckgo
        html = '<div class="result"><a class="result__a" href="https://example.com">T</a><div class="result__snippet">div snippet</div></div>'
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.web_search.httpx.get', return_value=mock_resp):
            results = _duckduckgo('q', 5)
        assert results[0]['snippet'] == 'div snippet'

    def test_duckduckgo_calls_httpx_correctly(self):
        from app.tools.web_search import _duckduckgo, DDG_URL, USER_AGENT
        mock_resp = MagicMock()
        mock_resp.text = ''
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.web_search.httpx.get', return_value=mock_resp) as mock_get:
            _duckduckgo('my query', 5)
            args, kwargs = mock_get.call_args
            assert args[0] == DDG_URL
            assert kwargs['params'] == {'q': 'my query'}
            assert kwargs['headers']['User-Agent'] == USER_AGENT
            assert kwargs['timeout'] == 15
            assert kwargs['follow_redirects'] is True

    def test_duckduckgo_raise_for_status_propagates(self):
        from app.tools.web_search import _duckduckgo
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError('404', request=MagicMock(), response=MagicMock())
        mock_resp.text = ''
        with patch('app.tools.web_search.httpx.get', return_value=mock_resp):
            with pytest.raises(httpx.HTTPStatusError):
                _duckduckgo('q', 5)

    def test_tavily_returns_snippet_from_content(self):
        from app.tools.web_search import _tavily
        mock_resp = MagicMock()
        mock_resp.json.return_value = {'results': [{'title': 'T', 'url': 'U', 'content': 'C'}]}
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.web_search.httpx.post', return_value=mock_resp):
            results = _tavily('q', 2)
        assert results[0]['snippet'] == 'C'

    def test_tavily_empty_results(self):
        from app.tools.web_search import _tavily
        mock_resp = MagicMock()
        mock_resp.json.return_value = {'results': []}
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.web_search.httpx.post', return_value=mock_resp):
            results = _tavily('q', 5)
        assert results == []

    def test_tavily_missing_results_key(self):
        from app.tools.web_search import _tavily
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.web_search.httpx.post', return_value=mock_resp):
            results = _tavily('q', 5)
        assert results == []

    def test_tavily_api_payload(self):
        from app.tools.web_search import _tavily
        mock_resp = MagicMock()
        mock_resp.json.return_value = {'results': []}
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.web_search.httpx.post', return_value=mock_resp) as mock_post, patch('app.tools.web_search.os.getenv', return_value='test_key_123'):
            _tavily('hello world', 3)
            kwargs = mock_post.call_args[1]
            assert kwargs['json']['api_key'] == 'test_key_123'
            assert kwargs['json']['query'] == 'hello world'
            assert kwargs['json']['max_results'] == 3
            assert kwargs['json']['search_depth'] == 'basic'
            assert kwargs['timeout'] == 20

    def test_tavily_handles_missing_fields(self):
        from app.tools.web_search import _tavily
        mock_resp = MagicMock()
        mock_resp.json.return_value = {'results': [{'url': 'U'}]}
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.web_search.httpx.post', return_value=mock_resp):
            results = _tavily('q', 5)
        assert results[0]['title'] == ''
        assert results[0]['url'] == 'U'
        assert results[0]['snippet'] == ''

class TestReadUrlExtended:

    def test_read_url_success(self):
        from app.tools.read_url import read_url
        html = '<html><body><p>Hello world</p></body></html>'
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.read_url.httpx.get', return_value=mock_resp):
            assert 'Hello world' in read_url('https://example.com')

    def test_read_url_strips_all_noise_tags(self):
        from app.tools.read_url import read_url
        html = '<script>js</script><style>css</style><nav>nav</nav><footer>foot</footer><header>head</header><noscript>ns</noscript><form>form</form><aside>aside</aside><p>keep</p>'
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.read_url.httpx.get', return_value=mock_resp):
            text = read_url('https://example.com')
            assert 'keep' in text
            for bad in ['js', 'css', 'nav', 'foot', 'head', 'ns', 'form', 'aside']:
                assert bad not in text

    def test_read_url_empty_body(self):
        from app.tools.read_url import read_url
        mock_resp = MagicMock()
        mock_resp.text = '<html><body>   \n\t  </body></html>'
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.read_url.httpx.get', return_value=mock_resp):
            assert 'No readable text' in read_url('https://example.com')

    def test_read_url_whitespace_collapse(self):
        from app.tools.read_url import read_url
        mock_resp = MagicMock()
        mock_resp.text = '<p>hello   \n\n  world\t\tfoo</p>'
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.read_url.httpx.get', return_value=mock_resp):
            assert read_url('https://example.com') == 'hello world foo'

    def test_read_url_truncates_default(self):
        from app.tools.read_url import read_url
        long = 'a' * 20000
        mock_resp = MagicMock()
        mock_resp.text = f'<p>{long}</p>'
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.read_url.httpx.get', return_value=mock_resp):
            text = read_url('https://example.com', max_chars=100)
            assert len(text) == 100
            assert text == 'a' * 100

    def test_read_url_max_chars_as_string(self):
        from app.tools.read_url import read_url
        mock_resp = MagicMock()
        mock_resp.text = '<p>' + 'a' * 1000 + '</p>'
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.read_url.httpx.get', return_value=mock_resp):
            text = read_url('https://example.com', max_chars='50')
            assert len(text) == 50

    def test_read_url_httpx_params(self):
        from app.tools.read_url import read_url, USER_AGENT
        mock_resp = MagicMock()
        mock_resp.text = '<p>hi</p>'
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.read_url.httpx.get', return_value=mock_resp) as mock_get:
            read_url('https://example.com/page')
            args, kwargs = mock_get.call_args
            assert args[0] == 'https://example.com/page'
            assert kwargs['headers']['User-Agent'] == USER_AGENT
            assert kwargs['timeout'] == 20
            assert kwargs['follow_redirects'] is True

    def test_read_url_exception_network(self):
        from app.tools.read_url import read_url
        with patch('app.tools.read_url.httpx.get', side_effect=RuntimeError('net fail')):
            text = read_url('https://example.com')
            assert 'Error reading https://example.com' in text
            assert 'RuntimeError' in text
            assert 'net fail' in text

    def test_read_url_exception_http_error(self):
        from app.tools.read_url import read_url
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError('404', request=MagicMock(), response=MagicMock())
        mock_resp.text = ''
        with patch('app.tools.read_url.httpx.get', return_value=mock_resp):
            text = read_url('https://example.com')
            assert 'Error reading' in text
            assert 'HTTPStatusError' in text

    def test_read_url_timeout_exception(self):
        from app.tools.read_url import read_url
        with patch('app.tools.read_url.httpx.get', side_effect=httpx.TimeoutException('timeout')):
            text = read_url('https://example.com')
            assert 'TimeoutException' in text

    def test_read_url_nested_tags(self):
        from app.tools.read_url import read_url
        html = '<html><body><div><p>Hello <b>bold</b> <i>italic</i></p></div></body></html>'
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        with patch('app.tools.read_url.httpx.get', return_value=mock_resp):
            text = read_url('https://example.com')
            assert 'Hello bold italic' in text

    def test_read_url_long_url_in_error(self):
        from app.tools.read_url import read_url
        long_url = 'https://example.com/' + 'a' * 100
        with patch('app.tools.read_url.httpx.get', side_effect=RuntimeError('fail')):
            text = read_url(long_url)
            assert long_url in text
