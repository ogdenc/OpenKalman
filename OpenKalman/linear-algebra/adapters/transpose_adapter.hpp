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
 * \brief Definition for \ref transpose_adapter.
 */

#ifndef OPENKALMAN_TRANSPOSE_ADAPTER_HPP
#define OPENKALMAN_TRANSPOSE_ADAPTER_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/adapters/internal/adapter_base.hpp"
#include "linear-algebra/interfaces/stl/mdspan-object.hpp"
#include "linear-algebra/interfaces/stl/transpose_mdspan_policies.hpp"

namespace OpenKalman
{
  /**
   * \brief An adapter that transposes a nested \ref indexible object.
   */
#ifdef __cpp_concepts
  template<indexible Nested, std::size_t indexa = 0, std::size_t indexb = 1> requires (indexa < indexb)
#else
  template<typename Nested, std::size_t indexa = 0, std::size_t indexb = 1>
#endif
  struct transpose_adapter : internal::adapter_base<transpose_adapter<Nested, indexa, indexb>, Nested>
  {
  private:

#ifndef __cpp_concepts
    static_assert(indexible<Nested>);
    static_assert(indexa < indexb);
#endif

    using Base = internal::adapter_base<transpose_adapter, Nested>;

  public:

    /**
     * \brief Construct from an \ref indexible object.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires std::constructible_from<Nested, Arg&&>
#else
    template<typename Arg, std::enable_if_t<stdex::constructible_from<Nested, Arg&&>, int> = 0>
#endif
    constexpr
    transpose_adapter(Arg&& arg) : Base {std::forward<Arg>(arg)}
    {}


    /**
     * \brief Default constructor.
     */
    constexpr transpose_adapter() = default;

  };


  /**
   * \brief Deduction guide
   */
#ifdef __cpp_concepts
  template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  transpose_adapter(Arg&&) -> transpose_adapter<Arg>;


  ///////////////////////
  //     interface     //
  ///////////////////////

  namespace interface
  {
    namespace detail
    {
      template<typename Nested>
      struct transpose_object_traits
      {
        static const bool is_specialized = true;

        static constexpr triangle_type
        triangle_type_value =
          triangle_type_of_v<Nested> == triangle_type::lower ? triangle_type::upper :
          triangle_type_of_v<Nested> == triangle_type::upper ? triangle_type::lower :
          triangle_type_of_v<Nested>;

        template<std::size_t N, applicability b>
        static constexpr bool
        is_square = square_shaped<Nested, N, b>;

        static constexpr bool
        is_hermitian = hermitian_matrix<Nested>;

      };
    }


    /**
     * \brief Interface traits for \ref transpose_adapter
     */
    template<typename Nested, std::size_t indexa, std::size_t indexb>
    struct object_traits<transpose_adapter<Nested, indexa, indexb>>
      : detail::transpose_object_traits<Nested>
    {
      static constexpr auto
      get_mdspan = [](auto&& t)
      {
        auto p = patterns::views::transpose<indexa, indexb>(
          OpenKalman::get_pattern_collection(std::forward<decltype(t)>(t).nested_object()));
        auto n = OpenKalman::get_mdspan(std::forward<decltype(t)>(t).nested_object());
        using N = std::decay_t<decltype(n)>;
        using nested_layout = typename N::layout_type;
        using extents_type = std::decay_t<decltype(to_extents(p))>;
        if constexpr (indexb == 1 and n.rank() == 2)
        {
          using layout_type = stdex::linalg::layout_transpose<nested_layout>;
          using mapping_type = typename layout_type::template mapping<extents_type>;
          return stdex::mdspan(
            n.data_handle(),
            mapping_type {n.mapping()},
            n.accessor());
        }
        else
        {
          using nested_mapping_type = typename N::mapping_type;
          using layout_type = interface::layout_transpose<nested_mapping_type, indexa, indexb>;
          using mapping_type = typename layout_type::template mapping<extents_type>;
          return stdex::mdspan(
            n.data_handle(),
            mapping_type {n.mapping(), to_extents(p)},
            n.accessor());
        }
      };


      static constexpr auto
      get_pattern_collection = [](auto&& t) -> decltype(auto)
      {
        return patterns::views::transpose<indexa, indexb>(
          OpenKalman::get_pattern_collection(std::forward<decltype(t)>(t).nested_object()));
      };


#ifdef __cpp_concepts
      template<typename T> requires get_constant_defined_for<Nested>
#else
      template<typename T, bool Enable = true, std::enable_if_t<Enable and get_constant_defined_for<Nested>, int> = 0>
#endif
      static constexpr auto
      get_constant(T&& t)
      {
        return object_traits<std::decay_t<Nested>>::get_constant(std::forward<T>(t).nested_object());
      }

    };


    template<typename N, typename E, typename NL, typename A>
    struct object_traits<stdex::mdspan<N, E, stdex::linalg::layout_transpose<NL>, A>>
      : detail::transpose_object_traits<stdex::mdspan<N, E, NL, A>>,
        internal::mdspan_base_object_traits<N, E, NL, A>
    {};


    /**
     * \brief Library interface traits for \ref transpose_adapter
     */
    template<typename Nested, std::size_t indexa, std::size_t indexb>
    struct library_interface<transpose_adapter<Nested, indexa, indexb>>
    {
    };

  }


}


#endif
