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
 * \brief Definitions for \ref to_diagonal_adapter
 */

#ifndef OPENKALMAN_TO_DIAGONAL_ADAPTER_HPP
#define OPENKALMAN_TO_DIAGONAL_ADAPTER_HPP

#include "linear-algebra/enumerations.hpp"
#include "linear-algebra/interfaces/interfaces-defined.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/concepts/pattern_collection_for.hpp"
#include "linear-algebra/traits/is_square_shaped.hpp"
#include "linear-algebra/traits/constant_value_of.hpp"
#include "linear-algebra/adapters/internal/adapter_base.hpp"
#include "linear-algebra/interfaces/stl/layout_to_diagonal.hpp"
#include "linear-algebra/interfaces/stl/to_diagonal_accessor.hpp"

namespace OpenKalman
{

#ifdef __cpp_concepts
  template<indexible Nested, patterns::pattern_collection PatternCollection> requires
    pattern_collection_for<decltype(patterns::views::diagonal_of(
      patterns::to_extents(std::declval<PatternCollection>()))), Nested>
#else
  template<typename Nested, typename PatternCollection>
#endif
  struct to_diagonal_adapter
    : OpenKalman::internal::adapter_base<to_diagonal_adapter<Nested, PatternCollection>, Nested>
  {

#ifndef __cpp_concepts
    static_assert(indexible<Nested>;
    static_assert(patterns::pattern_collection<PatternCollection>);
    static_assert(pattern_collection_for<decltype(patterns::views::diagonal_of(
      patterns::to_extents(std::declval<PatternCollection>()))), Nested>);
#endif

  private:

    using Base = OpenKalman::internal::adapter_base<to_diagonal_adapter, Nested>;

  public:

    /**
     * \brief Construct from the nested type.
     */
#ifdef __cpp_concepts
    template<indexible Arg, patterns::pattern_collection P> requires
      std::constructible_from<Nested, Arg&&> and
      std::constructible_from<PatternCollection, P&&>
#else
    template<typename Arg, typename P, std::enable_if_t<
      stdex::constructible_from<Nested, Arg&&> and
      stdex::constructible_from<PatternCollection, P&&>, int> = 0>
#endif
    constexpr
    to_diagonal_adapter(Arg&& arg, P&& p)
#ifdef __cpp_contracts
    pre (patterns::compare_pattern_collections(patterns::views::diagonal_of(to_extents(patt_)),
           to_extents(get_pattern_collection(this->nested_object()))))
#endif
      : Base {std::forward<Arg>(arg)}, patt_ {std::forward<P>(p)}
    {
#ifndef __cpp_contracts
      using D = decltype(patterns::views::diagonal_of(patterns::to_extents(std::declval<PatternCollection>())));
      if constexpr (not pattern_collection_for<D, Nested, applicability::guaranteed>)
      {
        assert(patterns::compare_pattern_collections(patterns::views::diagonal_of(to_extents(patt_)), patterns::to_extents(patt_)));
      }
#endif
    }


    /**
     * \brief Default constructor.
     */
    constexpr to_diagonal_adapter() = default;

  private:

    PatternCollection patt_;

    template<typename T>
    friend struct interface::object_traits;

  };


  // ------------------------------ //
  //        Deduction guide         //
  // ------------------------------ //

  /**
   * \brief Deduce to_diagonal_adapter Nested from its constructor argument.
   */
#ifdef __cpp_concepts
  template<indexible Arg, patterns::pattern_collection P>
#else
  template<typename Arg, typename P, std::enable_if_t<indexible<Arg> and patterns::pattern_collection<P>, int> = 0>
#endif
  explicit to_diagonal_adapter(Arg&&, P&&) -> to_diagonal_adapter<Arg, stdex::remove_cvref_t<P>>;


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    template<typename Nested, typename P>
    struct object_traits<to_diagonal_adapter<Nested, P>>
    {
      static const bool is_specialized = true;


      static constexpr auto
      get_mdspan = [](auto&& t)
      {
        auto n = OpenKalman::get_mdspan(std::forward<decltype(t)>(t).nested_object());
        using N = std::decay_t<decltype(n)>;
        using nested_extents_type = typename N::extents_type;
        using nested_layout_type = typename N::layout_type;
        using nested_accessor = typename N::accessor_type;
        auto nested_m = n.mapping();

        using extents_type = std::decay_t<decltype(patterns::to_extents(std::declval<P>()))>;
        using layout_type = layout_to_diagonal<nested_layout_type, nested_extents_type>;
        using mapping_type = typename layout_type::template mapping<extents_type>;
        std::size_t partition_offset = nested_m.required_span_size();

        using accessor_type = interface::to_diagonal_accessor<nested_accessor>;
        using data_handle_type = typename accessor_type::data_handle_type;

        return stdex::mdspan(
          data_handle_type {n.data_handle(), partition_offset},
          mapping_type {nested_m, patterns::to_extents(std::forward<decltype(t)>(t).patt_)},
          accessor_type {n.accessor()});
      };


      static constexpr auto
      get_pattern_collection = [](auto&& t) -> decltype(auto)
      {
        return std::forward<decltype(t)>(t).patt_;
      };


      static constexpr triangle_type
      triangle_type_value = triangle_type::diagonal;


#ifdef __cpp_concepts
      template<typename T> requires constant_object<Nested>
#else
      template<typename T, bool Enable = true, std::enable_if_t<Enable and constant_object<Nested>, int> = 0>
#endif
      static constexpr auto
      get_constant(T&& t)
      {
        return constant_value(std::forward<T>(t).nested_object());
      }

    };


    template<typename N, typename E, typename NL, typename NE, typename NA>
    struct object_traits<stdex::mdspan<N, E, layout_to_diagonal<NL, NE>, to_diagonal_accessor<NA>>>
      : internal::mdspan_base_object_traits<N, NE, NL, NA>
    {
      static const bool is_specialized = true;

      static constexpr triangle_type
      triangle_type_value = triangle_type::diagonal;
    };


  }

}



#endif
