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
 * \brief Definition for \ref conjugate_adapter.
 */

#ifndef OPENKALMAN_CONJUGATE_ADAPTER_HPP
#define OPENKALMAN_CONJUGATE_ADAPTER_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/adapters/internal/adapter_base.hpp"
#include "linear-algebra/interfaces/stl/mdspan-object.hpp"

namespace OpenKalman
{
  /**
   * \brief An adapter that conjugates a nested \ref indexible object.
   */
#ifdef __cpp_concepts
  template<indexible Nested>
#else
  template<typename Nested>
#endif
  struct conjugate_adapter : internal::adapter_base<conjugate_adapter<Nested>, Nested>
  {
  private:

#ifndef __cpp_concepts
    static_assert(indexible<Nested>);
#endif

    using Base = internal::adapter_base<conjugate_adapter, Nested>;

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
    conjugate_adapter(Arg&& arg) : Base {std::forward<Arg>(arg)}
    {}


    /**
     * \brief Default constructor.
     */
    constexpr conjugate_adapter() = default;

  };


  /**
   * \brief Deduction guide
   */
#ifdef __cpp_concepts
  template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  conjugate_adapter(Arg&&) -> conjugate_adapter<Arg>;


  ///////////////////////
  //     interface     //
  ///////////////////////

  namespace interface
  {
    namespace detail
    {
      template<typename Nested>
      struct conjugate_object_traits
      {
        static const bool is_specialized = true;

        static constexpr triangle_type
        triangle_type_value = triangle_type_of_v<Nested>;

        template<std::size_t N, applicability b>
        static constexpr bool
        is_square = square_shaped<Nested, N, b>;

        static constexpr bool
        is_hermitian = hermitian_matrix<Nested>;
      };

    }


    /**
     * \brief Interface traits for \ref conjugate_adapter
     */
    template<typename Nested>
    struct object_traits<conjugate_adapter<Nested>>
      : detail::conjugate_object_traits<stdex::remove_cvref_t<Nested>>
    {
      static constexpr auto
      get_mdspan = [](auto&& t)
      {
        auto n = OpenKalman::get_mdspan(std::forward<decltype(t)>(t).nested_object());
        using N = std::decay_t<decltype(n)>;
        using nested_accessor = typename N::accessor_type;
        using accessor_type = stdex::linalg::conjugated_accessor<nested_accessor>;
        return stdex::mdspan(
          n.data_handle(),
          n.mapping(),
          accessor_type(n.accessor()));
      };


      static constexpr auto
      get_pattern_collection = [](auto&& t) -> decltype(auto)
      {
        return OpenKalman::get_pattern_collection(std::forward<decltype(t)>(t).nested_object());
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


    template<typename N, typename E, typename L, typename NA>
    struct object_traits<stdex::mdspan<N, E, L, stdex::linalg::conjugated_accessor<NA>>>
      : detail::conjugate_object_traits<stdex::mdspan<N, E, L, NA>>,
        internal::mdspan_base_object_traits<N, E, L, NA>
    {};


    /**
     * \brief Library interface traits for \ref conjugate_adapter
     */
    template<typename Nested>
    struct library_interface<conjugate_adapter<Nested>>
    {
    };


  }


}


#endif
