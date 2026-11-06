#ifndef KYTY_COMMON_SHA256_H_
#define KYTY_COMMON_SHA256_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Common {

class Sha256 {
public:
	Sha256() { Reset(); }

	void Reset() {
		m_total_bytes = 0;
		m_buf_len = 0;
		m_state[0] = 0x6a09e667u;
		m_state[1] = 0xbb67ae85u;
		m_state[2] = 0x3c6ef372u;
		m_state[3] = 0xa54ff53au;
		m_state[4] = 0x510e527fu;
		m_state[5] = 0x9b05688cu;
		m_state[6] = 0x1f83d9abu;
		m_state[7] = 0x5be0cd19u;
	}

	void Update(const void* data, size_t len) {
		const auto* p = static_cast<const uint8_t*>(data);
		m_total_bytes += len;
		while (len > 0) {
			const size_t to_copy = std::min(len, size_t(64) - m_buf_len);
			std::memcpy(m_buffer.data() + m_buf_len, p, to_copy);
			m_buf_len += to_copy;
			p += to_copy;
			len -= to_copy;
			if (m_buf_len == 64) {
				Transform(m_buffer.data());
				m_buf_len = 0;
			}
		}
	}

	std::array<uint8_t, 32> Finalize() {
		const uint64_t total_bits = m_total_bytes * 8ull;
		m_buffer[m_buf_len++] = 0x80u;
		if (m_buf_len > 56) {
			std::memset(m_buffer.data() + m_buf_len, 0, 64 - m_buf_len);
			Transform(m_buffer.data());
			m_buf_len = 0;
		}
		std::memset(m_buffer.data() + m_buf_len, 0, 56 - m_buf_len);
		for (int i = 0; i < 8; ++i) {
			m_buffer[56 + i] = static_cast<uint8_t>((total_bits >> ((7 - i) * 8)) & 0xFFu);
		}
		Transform(m_buffer.data());

		std::array<uint8_t, 32> digest;
		for (size_t i = 0; i < 8; ++i) {
			digest[i * 4 + 0] = static_cast<uint8_t>((m_state[i] >> 24) & 0xFF);
			digest[i * 4 + 1] = static_cast<uint8_t>((m_state[i] >> 16) & 0xFF);
			digest[i * 4 + 2] = static_cast<uint8_t>((m_state[i] >> 8) & 0xFF);
			digest[i * 4 + 3] = static_cast<uint8_t>(m_state[i] & 0xFF);
		}
		return digest;
	}

	static std::string DigestToHex(const std::array<uint8_t, 32>& digest) {
		static const char hex_chars[] = "0123456789ABCDEF";
		std::string s;
		s.reserve(64);
		for (uint8_t b : digest) {
			s.push_back(hex_chars[(b >> 4) & 0x0F]);
			s.push_back(hex_chars[b & 0x0F]);
		}
		return s;
	}

	static std::string ComputeHex(std::span<const uint8_t> data) {
		Sha256 ctx;
		ctx.Update(data.data(), data.size());
		return DigestToHex(ctx.Finalize());
	}

	static std::string ComputeFileHex(const std::filesystem::path& path) {
		std::ifstream file(path, std::ios::binary);
		if (!file.is_open()) {
			return "";
		}
		Sha256 ctx;
		std::array<char, 65536> buf;
		while (file.read(buf.data(), buf.size()) || file.gcount() > 0) {
			ctx.Update(buf.data(), static_cast<size_t>(file.gcount()));
		}
		return DigestToHex(ctx.Finalize());
	}

	static std::string GetCurrentExecutableSha256() {
#if defined(_WIN32)
		std::array<wchar_t, 32768> name {};
		GetModuleFileNameW(nullptr, name.data(), static_cast<uint32_t>(name.size()));
		return ComputeFileHex(name.data());
#else
		return ComputeFileHex("/proc/self/exe");
#endif
	}

private:
	static uint32_t RoTr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

	void Transform(const uint8_t* block) {
		static const uint32_t K[64] = {
		    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
		    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
		    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
		    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
		    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
		    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
		    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
		    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffeu, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

		uint32_t W[64];
		for (int t = 0; t < 16; ++t) {
			W[t] = (static_cast<uint32_t>(block[t * 4 + 0]) << 24) |
			       (static_cast<uint32_t>(block[t * 4 + 1]) << 16) |
			       (static_cast<uint32_t>(block[t * 4 + 2]) << 8) |
			       static_cast<uint32_t>(block[t * 4 + 3]);
		}
		for (int t = 16; t < 64; ++t) {
			const uint32_t s0 = RoTr(W[t - 15], 7) ^ RoTr(W[t - 15], 18) ^ (W[t - 15] >> 3);
			const uint32_t s1 = RoTr(W[t - 2], 17) ^ RoTr(W[t - 2], 19) ^ (W[t - 2] >> 10);
			W[t] = W[t - 16] + s0 + W[t - 7] + s1;
		}

		uint32_t a = m_state[0];
		uint32_t b = m_state[1];
		uint32_t c = m_state[2];
		uint32_t d = m_state[3];
		uint32_t e = m_state[4];
		uint32_t f = m_state[5];
		uint32_t g = m_state[6];
		uint32_t h = m_state[7];

		for (int t = 0; t < 64; ++t) {
			const uint32_t S1 = RoTr(e, 6) ^ RoTr(e, 11) ^ RoTr(e, 25);
			const uint32_t ch = (e & f) ^ ((~e) & g);
			const uint32_t temp1 = h + S1 + ch + K[t] + W[t];
			const uint32_t S0 = RoTr(a, 2) ^ RoTr(a, 13) ^ RoTr(a, 22);
			const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
			const uint32_t temp2 = S0 + maj;

			h = g;
			g = f;
			f = e;
			e = d + temp1;
			d = c;
			c = b;
			b = a;
			a = temp1 + temp2;
		}

		m_state[0] += a;
		m_state[1] += b;
		m_state[2] += c;
		m_state[3] += d;
		m_state[4] += e;
		m_state[5] += f;
		m_state[6] += g;
		m_state[7] += h;
	}

	uint64_t m_total_bytes = 0;
	size_t m_buf_len = 0;
	std::array<uint32_t, 8> m_state {};
	std::array<uint8_t, 64> m_buffer {};
};

} // namespace Common

#endif /* KYTY_COMMON_SHA256_H_ */
