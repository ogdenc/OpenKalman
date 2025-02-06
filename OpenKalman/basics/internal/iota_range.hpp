/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref internal::iota_range.
 */

#ifndef OPENKALMAN_IOTA_RANGE_HPP
#define OPENKALMAN_IOTA_RANGE_HPP

#include <type_traits>
#ifdef __cpp_lib_ranges
#include <ranges>
#endif

namespace OpenKalman::internal
{
#ifndef __cpp_lib_ranges_iota
  struct IotaRange
  {
    struct Iterator
    {
      using iterator_category = std::random_access_iterator_tag;
      using difference_type = std::ptrdiff_t;
      using value_type = std::size_t;
      explicit constexpr Iterator(std::size_t p) : pos{p} {}
      constexpr Iterator() : pos{0_uz} {};
      constexpr Iterator(const Iterator& other) = default;
      constexpr Iterator(Iterator&& other) noexcept = default;
      constexpr Iterator& operator=(const Iterator& other) = default;
      constexpr Iterator& operator=(Iterator&& other) noexcept = default;
      constexpr value_type operator*() const { return pos; }
      constexpr auto& operator++() noexcept { ++pos; return *this; }
      constexpr auto operator++(int) noexcept { auto temp = *this; ++*this; return temp; }
      constexpr auto& operator--() noexcept { --pos; return *this; }
      constexpr auto operator--(int) noexcept { auto temp = *this; --*this; return temp; }
      constexpr auto& operator+=(const difference_type diff) noexcept { pos += diff; return *this; }
      constexpr auto& operator-=(const difference_type diff) noexcept { pos -= diff; return *this; }
      constexpr auto operator+(const difference_type diff) const noexcept { return Iterator {pos + diff}; }
      friend constexpr auto operator+(const difference_type diff, const Iterator& it) noexcept { return Iterator {diff + it.pos}; }
      constexpr auto operator-(const difference_type diff) const noexcept { return Iterator {pos - diff}; }
      constexpr value_type operator[](difference_type offset) const { return pos + offset; }
      constexpr bool operator==(const Iterator& other) const noexcept { return pos == other.pos; }
#ifdef __cpp_impl_three_way_comparison
      constexpr auto operator<=>(const Iterator& other) const noexcept { return pos <=> other.pos; }
#else
      constexpr bool operator!=(const Iterator& other) const noexcept { return pos != other.pos; }
      constexpr bool operator<(const Iterator& other) const noexcept { return pos < other.pos; }
      constexpr bool operator>(const Iterator& other) const noexcept { return pos > other.pos; }
      constexpr bool operator<=(const Iterator& other) const noexcept { return pos <= other.pos; }
      constexpr bool operator>=(const Iterator& other) const noexcept { return pos >= other.pos; }
#endif

    private:

      std::size_t pos;

    }; // struct Iterator

    explicit constexpr IotaRange(std::size_t start, std::size_t bound) : start {start}, bound {bound} {}
    [[nodiscard]] constexpr std::size_t size() const { return bound - start; }
    [[nodiscard]] constexpr auto begin() const { return Iterator {start}; }
    [[nodiscard]] constexpr auto end() const { return Iterator {bound}; }

  private:

    std::size_t start;
    std::size_t bound;

  }; // struct IotaRange


#endif


  /**
   * \internal
   * \brief Create a \ref internal::sized_random_access_range iota
   * \details Equivalent to <code>std::ranges::views::iota(start, bound)</code>,
   * which should be used if available and there are no compatibility concerns.
   */
  constexpr auto
  iota_range(std::size_t start, std::size_t bound)
  {
#ifdef __cpp_lib_ranges_iota
    return std::ranges::views::iota(start, bound);
#else
    return IotaRange {start, bound};
#endif
  };

} // OpenKalman::internal


#endif //OPENKALMAN_IOTA_RANGE_HPP
