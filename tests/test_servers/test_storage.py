"""Tests for the Storage MCP server."""

from unittest.mock import patch, MagicMock


def _mock_filesystem():
    """Create a mock FileSystem object."""
    fs = MagicMock()
    return fs


class TestListFiles:
    def test_returns_file_listing(self):
        mock_fs = _mock_filesystem()
        mock_fs.ls.return_value = [
            {"name": "/data/file1.nc", "type": "file", "size": 1024},
            {"name": "/data/subdir", "type": "directory", "size": 0},
        ]

        with patch(
            "oceanum_mcp.servers.storage.server.get_storage_filesystem",
            return_value=mock_fs,
        ):
            from oceanum_mcp.servers.storage.server import list_files

            result = list_files(path="/data")
            assert "file1.nc" in result
            assert "subdir" in result
            assert "file" in result
            assert "dir" in result

    def test_empty_directory(self):
        mock_fs = _mock_filesystem()
        mock_fs.ls.return_value = []

        with patch(
            "oceanum_mcp.servers.storage.server.get_storage_filesystem",
            return_value=mock_fs,
        ):
            from oceanum_mcp.servers.storage.server import list_files

            result = list_files(path="/empty")
            assert "Empty directory" in result


class TestFileExists:
    def test_exists(self):
        mock_fs = _mock_filesystem()
        mock_fs.exists.return_value = True

        with patch(
            "oceanum_mcp.servers.storage.server.get_storage_filesystem",
            return_value=mock_fs,
        ):
            from oceanum_mcp.servers.storage.server import file_exists

            result = file_exists("/data/file.nc")
            assert "EXISTS" in result

    def test_not_found(self):
        mock_fs = _mock_filesystem()
        mock_fs.exists.return_value = False

        with patch(
            "oceanum_mcp.servers.storage.server.get_storage_filesystem",
            return_value=mock_fs,
        ):
            from oceanum_mcp.servers.storage.server import file_exists

            result = file_exists("/data/missing.nc")
            assert "NOT FOUND" in result


class TestReadFile:
    def test_reads_small_file(self):
        mock_fs = _mock_filesystem()
        mock_fs.info.return_value = {"size": 100}
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read.return_value = "file content here"
        mock_fs.open.return_value = mock_file

        with patch(
            "oceanum_mcp.servers.storage.server.get_storage_filesystem",
            return_value=mock_fs,
        ):
            from oceanum_mcp.servers.storage.server import read_file

            result = read_file("/data/small.txt")
            assert result == "file content here"

    def test_rejects_large_file(self):
        mock_fs = _mock_filesystem()
        mock_fs.info.return_value = {"size": 2_000_000}

        with patch(
            "oceanum_mcp.servers.storage.server.get_storage_filesystem",
            return_value=mock_fs,
        ):
            from oceanum_mcp.servers.storage.server import read_file

            result = read_file("/data/large.bin")
            assert "too large" in result


class TestWriteFile:
    def test_writes_content(self):
        mock_fs = _mock_filesystem()
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_fs.open.return_value = mock_file

        with patch(
            "oceanum_mcp.servers.storage.server.get_storage_filesystem",
            return_value=mock_fs,
        ):
            from oceanum_mcp.servers.storage.server import write_file

            result = write_file("/data/output.txt", "hello world")
            assert "Written" in result
            assert "11 bytes" in result
            mock_file.write.assert_called_once_with("hello world")


class TestDeleteFile:
    def test_deletes_file(self):
        mock_fs = _mock_filesystem()

        with patch(
            "oceanum_mcp.servers.storage.server.get_storage_filesystem",
            return_value=mock_fs,
        ):
            from oceanum_mcp.servers.storage.server import delete_file

            result = delete_file("/data/old.nc")
            assert "Deleted" in result
            mock_fs.rm.assert_called_once_with("/data/old.nc", recursive=False)

    def test_deletes_recursive(self):
        mock_fs = _mock_filesystem()

        with patch(
            "oceanum_mcp.servers.storage.server.get_storage_filesystem",
            return_value=mock_fs,
        ):
            from oceanum_mcp.servers.storage.server import delete_file

            result = delete_file("/data/dir", recursive=True)
            mock_fs.rm.assert_called_once_with("/data/dir", recursive=True)


class TestFileInfo:
    def test_returns_info(self):
        mock_fs = _mock_filesystem()
        mock_fs.info.return_value = {
            "name": "/data/file.nc",
            "type": "file",
            "size": 4096,
        }

        with patch(
            "oceanum_mcp.servers.storage.server.get_storage_filesystem",
            return_value=mock_fs,
        ):
            from oceanum_mcp.servers.storage.server import file_info

            result = file_info("/data/file.nc")
            assert "/data/file.nc" in result
            assert "file" in result
            assert "4096" in result
