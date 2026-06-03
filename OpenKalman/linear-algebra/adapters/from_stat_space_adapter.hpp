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
 * \brief Definition for \ref from_stat_space_adapter.
 */

#ifndef OPENKALMAN_FROM_STAT_SPACE_ADAPTER_HPP
#define OPENKALMAN_FROM_STAT_SPACE_ADAPTER_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/concepts/pattern_collection_for.hpp"
#include "linear-algebra/concepts/constant_object.hpp"
#include "linear-algebra/traits/element_type_of.hpp"
#include "linear-algebra/traits/get_index_pattern.hpp"
#include "linear-algebra/views/range_of.hpp"
#include "linear-algebra/traits/internal/library_base.hpp"

namespace OpenKalman
{

  /**
   * \brief An adapter that transforms a patterned object back from a Euclidean space for directional statistics.
   * \details This is the counterpart expression to to_stat_space_adapter.
   * \note This should not usually be constructed directly. Instead, call \ref from_stat_space.
   * \tparam Nested The nested, pre-transformed object.
   * \tparam Pattern The pattern of index 0 of the result.
   */
#ifdef __cpp_concepts
  template<indexible Nested, patterns::pattern_collection PatternCollection> requires
    patterns::euclidean_pattern_collection<pattern_collection_type_of_t<Nested>> and
    pattern_collection_for<decltype(patterns::to_stat_space_pattern_collection(std::declval<PatternCollection>())), Nested> and
    std::same_as<PatternCollection, std::decay_t<PatternCollection>>
#else
  template<typename NestedObject, typename Pattern>
#endif
  struct from_stat_space_adapter
    : internal::library_base_t<to_stat_space_adapter<Nested>, Nested>
  {

#ifndef __cpp_concepts
    static_assert(patterns::euclidean_pattern_collection<pattern_collection_type_of_t<Nested>>);
    static_assert(pattern_collection_for<decltype(patterns::to_stat_space_pattern_collection(std::declval<PatternCollection>())), Nested>);
    static_assert(requires std::is_same_v<PatternCollection, std::decay_t<PatternCollection>>);
#endif

    using element_type = element_type_of_t<Nested>;

    using pattern_type = PatternCollection;

    using extents_type = decltype(patterns::to_extents(std::declval<pattern_type>()));

    using mdspan_type = stdex::mdspan<element_type, extents_type, stdex::layout_left>;


    template<std::size_t i = 0, typename Prod = std::integral_constant<std::size_t, 1>>
    static constexpr auto
    store_size(const Nested& n, Prod prod = {})
    {
      if constexpr (i < index_count_v<Nested>)
        return store_size<i + 1>(n,
          values::operation(
            std::multiplies{},
            std::move(prod),
            patterns::get_dimension(get_index_pattern<i>(n))
          ));
      else
        return prod;
    }


    using store_size_type = decltype(store_size(std::declval<Nested>()));

    using store_type = std::conditional_t<
      values::fixed<store_size_type>,
      std::array<element_type, values::fixed_value_of_v<store_size_type>>,
      std::vector<element_type>>;


    template<std::size_t N, std::size_t i = N, typename P, typename It, typename...J>
    static constexpr auto
    fill_store(const P& p, const Nested& nested, It it, J...j)
    {
      if constexpr (i == 0)
        it = stdex::ranges::copy(patterns::from_stat_space(p, collections::views::range_of(nested, j...)), std::move(it)).out;
      else for (std::size_t k = 0; k < get_index_extent<i>(nested); k++)
        it = fill_store<N, i - 1>(p, nested, std::move(it), k, j...);
      return it;
    }


    template<typename P>
    static constexpr store_type
    store_init(const Nested& nested, const P& p)
    {
      store_type store;
      if constexpr (not values::fixed<store_size_type>) store.reserve(store_size(nested));
      fill_store<index_count_v<Nested>>(p, nested, store.begin());
      return store;
    }

  public:

    /**
     * \brief Construct from an \ref indexible object and a \ref patterns::pattern_collection "pattern_collection".
     */
#ifdef __cpp_concepts
    template<patterns::pattern_collection P> requires
      std::constructible_from<PatternCollection, P&&>
#else
    template<typename P, std::enable_if_t<
      stdex::constructible_from<PatternCollection, P&&>, int> = 0>
#endif
    constexpr
    from_stat_space_adapter(const Nested& nested, P&& p)
      : pattern_ {std::forward<P>(p)},
        store_ {store_init(nested, patterns::get_pattern<0>(pattern_))},
        mdspan_ {store_.data(), stdex::layout_left::mapping {patterns::to_extents(pattern_)}}
    {}


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

    pattern_type pattern_;
    store_type store_;
    mdspan_type mdspan_;

    friend struct interface::object_traits<from_stat_space_adapter>;

  };


  /**
   * \brief Deduction guide
   */
#ifdef __cpp_concepts
  template<indexible Arg, patterns::pattern_collection P>
#else
  template<typename Arg, typename P, std::enable_if_t<indexible<Arg> and patterns::pattern_collection<P>, int> = 0>
#endif
  from_stat_space_adapter(Arg&&, const P&) -> from_stat_space_adapter<Arg, P>;


  namespace interface
  {
    template<typename Nested, typename PatternCollection>
    struct object_traits<from_stat_space_adapter<Nested, PatternCollection>>
    {
      static const bool is_specialized = true;

      static constexpr auto
      get_mdspan = [](auto&& t)
      {
        return std::forward<decltype(t)>(t).mdspan_;
      };


      static constexpr auto
      get_pattern_collection = [](auto&& t) -> decltype(auto)
      {
        return std::forward<decltype(t)>(t).pattern_;
      };

    };


    // --------------------- //
    //   library_interface   //
    // --------------------- //

    template<typename NestedObject, typename V0>
    struct library_interface<from_stat_space_adapter<NestedObject, V0>>
    {
      /*
    private:

      using NestedInterface = library_interface<NestedObject>;

    public:

      template<typename Derived>
      using library_base = internal::library_base_t<Derived, std::decay_t<NestedObject>>;


#ifdef __cpp_lib_ranges
      template<indexible Arg, std::ranges::input_range Indices> requires values::index<std::ranges::range_value_t<Indices>>
      static constexpr values::scalar decltype(auto)
#else
      template<typename Arg, typename Indices>
      static constexpr decltype(auto)
#endif
      get_component(Arg&& arg, const Indices& indices)
      {
        if constexpr (patterns::euclidean_pattern<V0>)
        {
          return NestedInterface::get_component(nested_object(std::forward<Arg>(arg)), indices);
        }
        else
        {
          auto g {[&arg, is...](std::size_t ix) { return OpenKalman::get_component(nested_object(std::forward<Arg>(arg)), ix, is...); }};
          if constexpr (to_euclidean_expr<nested_object_of_t<Arg>>)
            return patterns::wrap(get_pattern_collection<0>(arg), g, i);
          else
            return patterns::from_stat_space(get_pattern_collection<0>(arg), g, i);
        }
      }


#ifdef __cpp_lib_ranges
      template<indexible Arg, std::ranges::input_range Indices> requires values::index<std::ranges::range_value_t<Indices>>
#else
      template<typename Arg, typename Indices>
#endif
      static void
      set_component(Arg& arg, const scalar_type_of_t<Arg>& s, const Indices& indices)
      {
        if constexpr (patterns::euclidean_pattern<vector_space_descriptor_of<Arg, 0>>)
        {
          OpenKalman::set_component(nested_object(nested_object(arg)), s, indices);
        }
        else if constexpr (to_euclidean_expr<nested_object_of_t<Arg>>)
        {
          auto s {[&arg, is...](const scalar_type_of_t<Arg>& x, std::size_t i) {
            return OpenKalman::set_component(nested_object(nested_object(arg)), x, i, is...);
          }};
          auto g {[&arg, is...](std::size_t ix) {
            return OpenKalman::get_component(nested_object(nested_object(arg)), ix, is...);
          }};
          patterns::set_wrapped_component(get_pattern_collection<0>(arg), s, g, s, i);
        }
        else
        {
          OpenKalman::set_component(nested_object(arg), s, i, is...);
        }
      }


      template<typename Arg>
      static decltype(auto) to_native_matrix(Arg&& arg)
      {
        return OpenKalman::to_native_matrix<nested_object_of_t<Arg>>(std::forward<Arg>(arg));
      }


      template<data_layout layout, typename Scalar, typename D>
      static auto make_default(D&& d)
      {
        return make_dense_object<NestedObject, layout, Scalar>(std::forward<D>(d));
      }


      // fill_components not necessary because T is not a dense writable matrix.


      template<typename C, typename D>
      static constexpr auto make_constant(C&& c, D&& d)
      {
        return make_constant<NestedObject>(std::forward<C>(c), std::forward<D>(d));
      }


      template<typename Scalar, typename D>
      static constexpr auto make_identity_matrix(D&& d)
      {
        return make_identity_matrix_like<NestedObject, Scalar>(std::forward<D>(d));
      }


      // get_slice


      // set_slice


      template<typename Arg>
      static auto
      to_diagonal(Arg&& arg)
      {
        if constexpr( has_untyped_index<Arg, 0>)
        {
          return to_diagonal(nested_object(std::forward<Arg>(arg)));
        }
        else
        {
          return library_interface<NestedObject>::to_diagonal(to_native_matrix<NestedObject>(std::forward<Arg>(arg)));
        }
      }


      template<typename Arg>
      static auto
      diagonal_of(Arg&& arg)
      {
        if constexpr(has_untyped_index<Arg, 0>)
        {
          return diagonal_of(nested_object(std::forward<Arg>(arg)));
        }
        else
        {
          return library_interface<NestedObject>::diagonal_of(to_native_matrix<NestedObject>(std::forward<Arg>(arg)));
        }
      }


      template<typename Arg, typename...Factors>
      static auto
      broadcast(Arg&& arg, const Factors&...factors)
      {
        return library_interface<std::decay_t<nested_object_of_t<Arg>>>::broadcast(std::forward<Arg>(arg), factors...);
      }


      template<typename...Ds, typename Operation, typename...Args>
      static constexpr decltype(auto)
      n_ary_operation(const std::tuple<Ds...>& tup, Operation&& op, Args&&...args)
      {
        return library_interface<NestedObject>::template n_ary_operation(tup, std::forward<Operation>(op), std::forward<Args>(args)...);
      }


      template<std::size_t...indices, typename BinaryFunction, typename Arg>
      static constexpr decltype(auto)
      reduce(BinaryFunction&& b, Arg&& arg)
      {
        return library_interface<NestedObject>::template reduce<indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg));
      }


      template<typename Arg>
      constexpr decltype(auto)
      to_stat_space(Arg&& arg)
      {
        return nested_object(std::forward<Arg>(arg)); //< from- and then to- is an identity.
      }


      // from_stat_space not included. Double application of from_stat_space does not make sense.


      template<typename Arg>
      constexpr decltype(auto)
      wrap_angles(Arg&& arg)
      {
        return std::forward<Arg>(arg); //< A from_stat_space_adapter is already wrapped
      }


      template<typename Arg>
      static constexpr decltype(auto)
      conjugate(Arg&& arg)
      {
        if constexpr(has_untyped_index<Arg, 0>)
        {
          return OpenKalman::conjugate(nested_object(std::forward<Arg>(arg)));
        }
        else
        {
          return std::forward<Arg>(arg).conjugate(); //< \todo Generalize this.
        }
      }


      template<typename Arg>
      static constexpr decltype(auto)
      transpose(Arg&& arg)
      {
        if constexpr(has_untyped_index<Arg, 0>)
        {
          return OpenKalman::transpose(nested_object(std::forward<Arg>(arg)));
        }
        else
        {
          return std::forward<Arg>(arg).transpose(); //< \todo Generalize this.
        }
      }


      template<typename Arg>
      static constexpr decltype(auto)
      conjugate_transpose(Arg&& arg)
      {
        if constexpr(has_untyped_index<Arg, 0>)
        {
          return OpenKalman::conjugate_transpose(nested_object(std::forward<Arg>(arg)));
        }
        else
        {
          return std::forward<Arg>(arg).conjugate_transpose(); //< \todo Generalize this.
        }
      }


      template<typename Arg>
      static constexpr auto
      determinant(Arg&& arg)
      {
        if constexpr(has_untyped_index<Arg, 0>)
        {
          return OpenKalman::determinant(nested_object(std::forward<Arg>(arg)));
        }
        else
        {
          return arg.determinant(); //< \todo Generalize this.
        }
      }


      template<triangle_type significant_triangle, typename A, typename U, typename Alpha>
      static decltype(auto)
      rank_update_hermitian(A&& a, U&& u, const Alpha alpha)
      {
        return OpenKalman::rank_update_hermitian<significant_triangle>(to_hermitian(to_dense_object(std::forward<A>(a))), std::forward<U>(u), alpha);
      }


      template<triangle_type triangle, typename A, typename U, typename Alpha>
      static decltype(auto) rank_update_triangular(A&& a, U&& u, const Alpha alpha)
      {
        return OpenKalman::rank_update_triangular(to_triangular<triangle>(to_dense_object(std::forward<A>(a))), std::forward<U>(u), alpha);
      }


      template<bool must_be_unique, bool must_be_exact, typename A, typename B>
      static constexpr decltype(auto)
      solve(A&& a, B&& b)
      {
        return OpenKalman::solve<must_be_unique, must_be_exact>(
          to_native_matrix<T>(std::forward<A>(a)), std::forward<B>(b));
      }


      template<typename A>
      static inline auto
      LQ_decomposition(A&& a)
      {
        return LQ_decomposition(to_dense_object(std::forward<A>(a)));
      }


      template<typename A>
      static inline auto
      QR_decomposition(A&& a)
      {
        return QR_decomposition(to_dense_object(std::forward<A>(a)));
      }

*/
    };


  }


}


#endif
