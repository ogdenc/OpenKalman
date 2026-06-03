/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref pattern_adapter.
 */

#ifndef OPENKALMAN_PATTERN_ADAPTER_HPP
#define OPENKALMAN_PATTERN_ADAPTER_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/concepts/pattern_collection_for.hpp"
#include "linear-algebra/adapters/internal/adapter_base.hpp"
#include "linear-algebra/concepts/square_shaped.hpp"
#include "linear-algebra/concepts/hermitian_matrix.hpp"

namespace OpenKalman
{
  /**
   * \brief An adapter that attaches a \ref patterns::pattern_collection "pattern_collection" to an \ref indexible object.
   * \details Any vector space descriptors associated with the nested object are effectively overwritten.
   * The adapter can be owning or non-owning, depending on whether Nested is an lvalue reference.
   * \note This should not normally be constructed directly. Instead, call \ref attach_patterns.
   */
#ifdef __cpp_concepts
  template<indexible Nested, pattern_collection_for<Nested> PatternCollection>
  requires std::same_as<PatternCollection, std::decay_t<PatternCollection>>
#else
  template<typename Nested, typename PatternCollection>
#endif
  struct pattern_adapter : internal::adapter_base<pattern_adapter<Nested, PatternCollection>, Nested>
  {
  private:

#ifndef __cpp_concepts
    static_assert(pattern_collection_for<PatternCollection, Nested>);
    static_assert(requires std::is_same_v<PatternCollection, std::decay_t<PatternCollection>>);
#endif

    using Base = internal::adapter_base<pattern_adapter, Nested>;

  public:

    /**
     * \brief Construct from an \ref indexible object and a \ref patterns::pattern_collection "pattern_collection".
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
    pattern_adapter(Arg&& arg, P&& p)
#ifdef __cpp_contracts
      pre(pattern_collection_for<PatternCollection, Nested, applicability::guaranteed> or
        not patterns::compare_pattern_collections(to_extents(pattern_), get_pattern_collection(this->nested_object())))
#endif
      : Base {std::forward<Arg>(arg)}, pattern_ {std::forward<P>(p)}
    {
#ifndef __cpp_contracts
      if constexpr(not pattern_collection_for<PatternCollection, Nested, applicability::guaranteed>)
      {
        assert(not patterns::compare_pattern_collections(to_extents(pattern_), get_pattern_collection(this->nested_object())));
      }
#endif
    }


    /**
     * \brief Default constructor.
     */
    constexpr pattern_adapter() = default;


    /**
     * \brief Get the associated \ref patterns::pattern_collection "pattern_collection".
     */
#ifdef __cpp_explicit_this_parameter
    template<typename Self>
    constexpr decltype(auto)
    pattern_collection(this Self&& self)
    {
      return std::forward<Self>(self).pattern_;
    }
#else
    constexpr PatternCollection&
    pattern_collection() & { return pattern_; }

    /// \overload
    constexpr const PatternCollection&
    pattern_collection() const & { return pattern_; }

    /// \overload
    constexpr PatternCollection&&
    pattern_collection() && { return std::move(*this).pattern_; }

    /// \overload
    constexpr const PatternCollection&&
    pattern_collection() const && { return std::move(*this).pattern_; }
#endif

  private:

    PatternCollection pattern_;

    friend struct interface::object_traits<pattern_adapter>;

  };


  /**
   * \brief Deduction guide
   */
#ifdef __cpp_concepts
  template<indexible Arg, pattern_collection_for<Arg> P>
#else
  template<typename Arg, typename P, std::enable_if_t<indexible<Arg> and pattern_collection_for<P, Arg>, int> = 0>
#endif
  pattern_adapter(Arg&&, P&&) -> pattern_adapter<Arg, stdex::remove_cvref_t<P>>;


  ///////////////////////
  //     interface     //
  ///////////////////////

  namespace interface
  {
    /**
     * \brief Interface traits for \ref pattern_adapter
     */
    template<typename Nested, typename PatternCollection>
    struct object_traits<pattern_adapter<Nested, PatternCollection>>
    {
      static const bool is_specialized = true;


      static constexpr auto
      get_mdspan = [](auto&& t)
      {
        return OpenKalman::get_mdspan(std::forward<decltype(t)>(t).nested_object());
      };


      static constexpr auto
      get_pattern_collection = [](auto&& t) -> decltype(auto)
      {
        return std::forward<decltype(t)>(t).pattern_;
      };


      static constexpr triangle_type
      triangle_type_value = triangle_type_of_v<Nested>;


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


      template<std::size_t N, applicability b>
      static constexpr bool
      is_square = square_shaped<Nested, N, b>;


      static constexpr bool
      is_hermitian = hermitian_matrix<Nested>;

    };


  }


}


#endif
