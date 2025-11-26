/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of transpose function.
 */

#ifndef OPENKALMAN_TRANSPOSE_HPP
#define OPENKALMAN_TRANSPOSE_HPP

#include "values/values.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/get_mdspan.hpp"
#include "linear-algebra/traits/element_type_of.hpp"
#include "linear-algebra/concepts/diagonal_matrix.hpp"
#include "linear-algebra/concepts/hermitian_matrix.hpp"
#include "linear-algebra/concepts/constant_object.hpp"
#include "conjugate.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<typename C, typename Arg, std::size_t...Is>
    constexpr decltype(auto) transpose_constant(C&& c, Arg&& arg, std::index_sequence<Is...>)
    {
      return make_constant<Arg>(std::forward<C>(c),
        get_pattern_collection<1>(arg), get_pattern_collection<0>(arg), get_pattern_collection<Is + 2>(arg)...);
    }


    template<typename NestedLayout, std::size_t indexa, std::size_t indexb>
    struct transpose_layout
    {
      template<class Extents>
      struct mapping {
        using extents_type = Extents;
        using index_type = typename extents_type::index_type;
        using size_type = typename extents_type::size_type;
        using rank_type = typename extents_type::rank_type;
        using layout_type = transpose_layout;

      private:

        template<typename = std::make_index_sequence<Extents::rank()>>
        struct transposed_extents {};

        template<std::size_t...i>
        struct transposed_extents<std::index_sequence<i...>>
        {
          using type = stdex::extents<index_type, (Extents::static_extent(
            i == indexa ? indexb : i == indexb ? indexa : i))...>;
        };

        template<std::size_t...i>
        constexpr typename transposed_extents<Extents>::type
        transpose_extents(const stdex::extents<index_type, i...>& e)
        {
          return {e.extent(i == indexa ? indexb : i == indexb ? indexa : i)...};
        }

        using nested_mapping_type = typename NestedLayout::template mapping<typename transposed_extents<Extents>::type>;

      public:

        constexpr explicit
        mapping(const nested_mapping_type& map)
          : nested_mapping_(map), extents_(transpose_extents(map.extents()))
        {}

        constexpr const extents_type&
        extents() const noexcept
        {
          return extents_;
        }

#ifdef __cpp_concepts
        template<std::convertible_to<index_type> IndexType0, std::convertible_to<index_type> IndexType1,
          std::convertible_to<index_type>...IndexTypes>
#else
        template<typename IndexType0, typename IndexType1, typename...IndexTypes, std::enable_if_t<
          std::is-convertible_v<IndexType0, index_type> and std::is-convertible_v<IndexType1, index_type> and
          (... and std::is_convertible_v<IndexTypes, index_type>), int> = 0>
#endif
        index_type
        operator() (IndexType0 i, IndexType1 j, IndexTypes...ks) const
        {
          return nested_mapping_(j, i, ks...);
        }

        constexpr index_type
        required_span_size() const noexcept(noexcept(nested_mapping_.required_span_size()))
        {
          return nested_mapping_.required_span_size();
        }

        const nested_mapping_type&
        nested_mapping() const { return nested_mapping_; }

        static constexpr bool
        is_always_unique() noexcept { return nested_mapping_type::is_always_unique(); }

        static constexpr bool
        is_always_exhaustive() noexcept { return nested_mapping_type::is_always_contiguous(); }

        static constexpr bool
        is_always_strided() noexcept { return nested_mapping_type::is_always_strided(); }

        constexpr bool
        is_unique() const { return nested_mapping_.is_unique(); }

        constexpr bool
        is_exhaustive() const { return nested_mapping_.is_exhaustive(); }

        constexpr bool
        is_strided() const { return nested_mapping_.is_strided(); }

        constexpr index_type
        stride(size_t r) const
        {
          assert(this->is_strided());
          assert(r < extents_type::rank());
          return nested_mapping_.stride(r == indexa ? indexb : r == indexb ? indexa : r);
        }

        template<class OtherExtents>
        friend constexpr bool
        operator==(const mapping& lhs, const mapping<OtherExtents>& rhs) noexcept
        {
          return lhs.nested_mapping_ == rhs.nested_mapping_;
        }

      private:

        nested_mapping_type nested_mapping_;
        extents_type extents_;

      };
    };


  }


  /**
   * \brief Swap any two indices of an \ref indexible_object
   * \details By default, the first two indices are transposed.
   */
#ifdef __cpp_concepts
  template<std::size_t indexa = 0, std::size_t indexb = 1, indexible Arg> requires (indexa < indexb)
#else
  template<std::size_t indexa = 0, std::size_t indexb = 1, typename Arg, std::enable_if_t<
    indexible<Arg> and (indexa < indexb), int> = 0>
#endif
  constexpr decltype(auto) transpose(Arg&& arg)
  {
    constexpr bool square_matrix = values::size_compares_with<index_dimension_of<Arg, 0>, index_dimension_of<Arg, 1>>;
    constexpr bool diag_invariant = (diagonal_matrix<Arg> or constant_object<Arg>) and square_matrix;
    constexpr bool hermitian_invariant = hermitian_matrix<Arg> and not values::complex<element_type_of_t<Arg>>;
    constexpr bool invariant = diag_invariant or hermitian_invariant;

    if constexpr (invariant)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (indexb == 1 and interface::matrix_transpose_defined_for<Arg&&>)
    {
      return interface::library_interface<stdex::remove_cvref_t<Arg>>::transpose(std::forward<Arg>(arg));
    }
    else if constexpr (interface::transpose_defined_for<Arg&&, indexa, indexb>)
    {
      return interface::library_interface<stdex::remove_cvref_t<Arg>>::template transpose<indexa, indexb>(std::forward<Arg>(arg));
    }
    else if constexpr (std::is_lvalue_reference_v<Arg> and index_count_v<Arg> <= 2)
    {
      return transpose(get_mdspan(arg));
    }
    else if constexpr (constant_object<Arg>)
    {
      constexpr std::make_index_sequence<std::max({index_count_v<Arg>, 2_uz}) - 2_uz> seq;
      return detail::transpose_constant(constant_value(arg), std::forward<Arg>(arg), seq);
    }
    else if constexpr (hermitian_matrix<Arg>)
    {
      return conjugate(std::forward<Arg>(arg));
    }
    else if constexpr (std::is_lvalue_reference_v<Arg>)
    {
      auto m = get_mdspan(arg);
      using layout_type = detail::transpose_layout<typename std::decay_t<decltype(m)>::layout_type, indexa, indexb>;

      auto mapping = layout_type::mapping(m.mapping());
      auto n = stdex::mdspan(m.data_handle(), mapping, m.accessor());
      return attach_pattern(std::move(n), get_pattern_collection(std::forward<Arg>(arg)));
    }
    else if (indexb == 1)
    {
      static_assert(interface::matrix_transpose_defined_for<Arg&&>, "Interface not defined");
    }
    else
    {
      static_assert(interface::transpose_defined_for<Arg&&, indexa, indexb>, "Interface not defined");
    }
  }


}

#endif
