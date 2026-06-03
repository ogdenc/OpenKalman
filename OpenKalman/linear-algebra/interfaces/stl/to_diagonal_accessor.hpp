/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief definition of \ref to_diagonal_accessor (an mdspan accessor)
 */

#ifndef OPENKALMAN_TO_DIAGONAL_ACCESSOR_HPP
#define OPENKALMAN_TO_DIAGONAL_ACCESSOR_HPP

#include "patterns/patterns.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief An accessor that returns either the diagonal element or 0.
   * \details This works in conjunction with \ref layout_triangle_partition.
   * It assumes that the element space is partitioned into diagonal, main triangle, and secondary triangle, respectively,
   * each partition being a translated copy of the nested accessor's accessible range and the secondary triangle transposed.
   */
  template<typename NestedAccessor>
  struct to_diagonal_accessor
  {
    using element_type = values::value_type_of_t<typename NestedAccessor::element_type>;
    using reference = element_type;

    /// A tuple of the nested data_handle_type and the partition offset.
    using data_handle_type = std::tuple<typename NestedAccessor::data_handle_type, std::size_t>;

    using offset_policy = to_diagonal_accessor;

    static_assert(values::value<element_type>);
#ifdef __cpp_concepts
    static_assert(requires { static_cast<reference>(std::declval<int>()); });
    static_assert(requires { static_cast<reference>(std::declval<typename NestedAccessor::reference>()); });
#endif

    /**
     * \brief
     * \param acc The nested accessor
     */
    constexpr
    to_diagonal_accessor(NestedAccessor acc) : nested_accessor_(std::move(acc)) {}

    to_diagonal_accessor() = delete;

#ifdef __cpp_concepts
    template<stdex::convertible_to<NestedAccessor> OtherNestedAccessor> requires
      (not std::is_same_v<NestedAccessor, OtherNestedAccessor>)
#else
    template<typename OtherNestedAccessor, std::enable_if_t<
      stdex::convertible_to<OtherNestedAccessor, NestedAccessor> and
      (not std::is_same_v<NestedAccessor, OtherNestedAccessor>), int> = 0>
#endif
    constexpr to_diagonal_accessor(const to_diagonal_accessor<OtherNestedAccessor>& other) noexcept
      : nested_accessor_ {other.element_} {}

#ifdef __cpp_concepts
    template<stdex::convertible_to<NestedAccessor> OtherNestedAccessor> requires
      (not std::is_same_v<NestedAccessor, OtherNestedAccessor>)
#else
    template<typename OtherNestedAccessor, std::enable_if_t<
      stdex::convertible_to<OtherNestedAccessor, NestedAccessor> and
      (not std::is_same_v<NestedAccessor, OtherNestedAccessor>), int> = 0>
#endif
    constexpr to_diagonal_accessor(to_diagonal_accessor<OtherNestedAccessor>&& other) noexcept
      : nested_accessor_ {std::move(other).element_} {}

    constexpr reference
    access(data_handle_type p, std::size_t i) const noexcept
    {
      auto [nested_p, partition_offset] = std::move(p);
      if (i < partition_offset) return static_cast<reference>(nested_accessor_.access(std::move(nested_p), i));
      return static_cast<reference>(0);
    }

    constexpr data_handle_type
    offset(data_handle_type p, std::size_t i) const noexcept
    {
      if (i == 0) return std::move(p);
      auto [nested_p, partition_offset] = std::move(p);
      return {std::move(nested_p), std::move(partition_offset) - i};
    }

    const NestedAccessor&
    nested_accessor() const noexcept { return nested_accessor_; }

  private:

    NestedAccessor nested_accessor_;

  };


}

#endif
