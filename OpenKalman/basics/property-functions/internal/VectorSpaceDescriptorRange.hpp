/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref VectorSpaceDescriptorRange.
 */

#ifndef OPENKALMAN_VECTORSPACEDESCRIPTORRANGE_HPP
#define OPENKALMAN_VECTORSPACEDESCRIPTORRANGE_HPP


namespace OpenKalman::internal
{
  template<typename Indexible>
  struct VectorSpaceDescriptorRange
  {
    struct Iterator
    {
      using difference_type = std::ptrdiff_t;
      using value_type = std::decay_t<decltype(get_vector_space_descriptor<0>(std::declval<Indexible>()))>; 
      static_assert(dynamic_vector_space_descriptor<value_type>);
      constexpr Iterator(const Indexible& indexible, const std::size_t p) : my_indexible{indexible}, pos{p} {}
      constexpr value_type operator*() const { return get_vector_space_descriptor(my_indexible, pos); } 
      constexpr auto& operator++() noexcept { ++pos; return *this; }
      constexpr auto operator++(int) noexcept { auto temp = *this; ++*this; return temp; }
      constexpr auto& operator--() noexcept { --pos; return *this; }
      constexpr auto operator--(int) noexcept { auto temp = *this; --*this; return temp; }
      constexpr auto& operator+=(const difference_type diff) noexcept { pos += diff; return *this; }
      constexpr auto& operator-=(const difference_type diff) noexcept { pos -= diff; return *this; }
      constexpr auto operator+(const difference_type diff) const noexcept { return Iterator {pos + diff}; }
      constexpr auto operator-(const difference_type diff) const noexcept { return Iterator {pos - diff}; }
      constexpr auto operator+(const Iterator& other) const noexcept { return Iterator {pos + other.pos}; }
      constexpr difference_type operator-(const Iterator& other) const noexcept { return pos - other.pos; }
      constexpr value_type operator[](difference_type offset) const { return get_vector_space_descriptor(my_indexible, pos + offset); }
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

      const Indexible& my_indexible;

      std::size_t pos;
      
    }; // struct Iterator

    constexpr VectorSpaceDescriptorRange(const VectorSpaceDescriptorRange& other) : my_indexible {other.my_indexible} {}
    
    constexpr VectorSpaceDescriptorRange(VectorSpaceDescriptorRange&& other) : my_indexible {other.my_indexible} {}
    
    constexpr VectorSpaceDescriptorRange(const Indexible& indexible) : my_indexible {indexible} {}
    
    constexpr auto begin() const { return Iterator {my_indexible, 0}; }
    
    constexpr auto end() const { return Iterator {my_indexible, count_indices(my_indexible)}; }
    
    constexpr std::size_t size() const { return count_indices(my_indexible); } 

  private:

    const Indexible& my_indexible;
    
  };
  
} // namespace OpenKalman::internal

#endif //OPENKALMAN_VECTORSPACEDESCRIPTORRANGE_HPP
