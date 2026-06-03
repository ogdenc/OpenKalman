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
 * \brief Definition for \ref to_stat_space_adapter.
 */

#ifndef OPENKALMAN_TO_STAT_SPACE_ADAPTER_HPP
#define OPENKALMAN_TO_STAT_SPACE_ADAPTER_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/concepts/constant_object.hpp"
#include "linear-algebra/traits/element_type_of.hpp"
#include "linear-algebra/traits/get_index_pattern.hpp"
#include "linear-algebra/views/range_of.hpp"
#include "linear-algebra/traits/internal/library_base.hpp"

namespace OpenKalman
{

  /**
   * \brief An adapter that transforms a patterned object into a Euclidean space for directional statistics.
   * \details This is the counterpart expression to from_stat_space_adapter.
   * \note This should not usually be constructed directly. Instead, call \ref to_stat_space.
   * \tparam Nested The nested, pre-transformed object.
   */
#ifdef __cpp_concepts
  template<indexible Nested> requires (not std::is_rvalue_reference_v<Nested>)
#else
  template<typename Nested>
#endif
  struct to_stat_space_adapter
    : internal::library_base_t<to_stat_space_adapter<Nested>, Nested>
  {
  private:

#ifndef __cpp_concepts
    static_assert(indexible<Nested>);
    static_assert(not std::is_rvalue_reference_v<Nested>);
#endif


    using element_type = element_type_of_t<Nested>;

    using extents_type = decltype(patterns::to_extents(patterns::to_stat_space_pattern_collection(std::declval<pattern_collection_type_of_t<Nested>>())));

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
            patterns::get_stat_dimension(get_index_pattern<i>(n))
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
        it = stdex::ranges::copy(patterns::to_stat_space(p, collections::views::range_of(nested, j...)), std::move(it)).out;
      else for (std::size_t k = 0; k < get_index_extent<i>(nested); k++)
        it = fill_store<N, i - 1>(p, nested, std::move(it), k, j...);
      return it;
    }


    static constexpr store_type
    store_init(const Nested& nested)
    {
      store_type store;
      auto p = get_index_pattern<0>(nested);
      if constexpr (not values::fixed<store_size_type>) store.reserve(store_size(nested));
      fill_store<index_count_v<Nested>>(p, nested, store.begin());
      return store;
    }

  public:

    /**
     * \brief Construct from an \ref indexible object and a \ref patterns::pattern_collection "pattern_collection".
     */
    constexpr
    to_stat_space_adapter(const Nested& nested)
      : store_ {store_init(nested)},
        mdspan_ {store_.data(), stdex::layout_left::mapping {
          patterns::to_extents(patterns::to_stat_space_pattern_collection(get_pattern_collection(nested)))}}
    {}

  private:

    store_type store_;
    mdspan_type mdspan_;

    friend struct interface::object_traits<to_stat_space_adapter>;

  };


  /**
   * \brief Deduction guide
   */
#ifdef __cpp_concepts
template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  to_stat_space_adapter(Arg&&) -> to_stat_space_adapter<Arg>;


  namespace interface
  {
    template<typename Nested>
    struct object_traits<to_stat_space_adapter<Nested>>
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
        return std::forward<decltype(t)>(t).mdspan_.extents();
      };

    };


    template<typename Nested>
    struct library_interface<to_stat_space_adapter<Nested>>
    {
/*    private:

        using NestedInterface = library_interface<Nested>;

    public:

      template<typename Derived>
      using library_base = internal::library_base_t<Derived, std::decay_t<Nested>>;


#ifdef __cpp_lib_ranges
      template<indexible Arg, std::ranges::input_range Indices> requires values::index<std::ranges::range_value_t<Indices>>
      static constexpr values::scalar decltype(auto)
#else
      template<typename Arg, typename Indices>
      static constexpr decltype(auto)
#endif
      access(Arg&& arg, const Indices& indices)
      {
        if constexpr (has_untyped_index<Nested, 0>)
        {
          return NestedInterface::access(nested_object(std::forward<Arg>(arg)), indices);
        }
        else
        {
          auto g {[&arg, is...](std::size_t ix) { return access(nested_object(std::forward<Arg>(arg)), ix, is...); }};
          return patterns::to_stat_space(get_pattern_collection<0>(arg), g, i);
        }
      }


#ifdef __cpp_lib_ranges
      template<indexible Arg, std::ranges::input_range Indices> requires values::index<std::ranges::range_value_t<Indices>>
#else
      template<typename Arg, typename Indices>
#endif
      static void
      set_component(Arg& arg, const element_type_of_t<Arg>& s, const Indices& indices)
      {
        if constexpr (has_untyped_index<Nested, 0>)
        {
          NestedInterface::set_component(nested_object(arg), s, indices);
        }
        else
        {
          set_component(nested_object(arg), s, indices);
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
        return make_dense_object<Nested, layout, Scalar>(std::forward<D>(d));
      }


      // fill_components not necessary because T is not a dense writable matrix.


      template<typename C, typename D>
      static constexpr auto make_constant(C&& c, D&& d)
      {
        return make_constant<Nested>(std::forward<C>(c), std::forward<D>(d));
      }


      template<typename Scalar, typename D>
      static constexpr auto make_identity_matrix(D&& d)
      {
        return make_identity_matrix_like<Nested, Scalar>(std::forward<D>(d));
      }


      // get_slice


      // set_slice


      template<typename Arg>
      static auto
      to_diagonal(Arg&& arg)
      {
        if constexpr( has_untyped_index<Nested, 0>)
        {
          return to_diagonal(nested_object(std::forward<Arg>(arg)));
        }
        else
        {
          return library_interface<P>::to_diagonal(to_native_matrix<Nested>(std::forward<Arg>(arg)));
        }
      }


      template<typename Arg>
      static auto
      diagonal_of(Arg&& arg)
      {
        if constexpr(has_untyped_index<Nested, 0>)
        {
          return diagonal_of(nested_object(std::forward<Arg>(arg)));
        }
        else
        {
          return library_interface<P>::diagonal_of(to_native_matrix<Nested>(std::forward<Arg>(arg)));
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
        using P = std::decay_t<Nested>;
        return library_interface<P>::template n_ary_operation(tup, std::forward<Operation>(op), std::forward<Args>(args)...);
      }


      template<std::size_t...indices, typename BinaryFunction, typename Arg>
      static constexpr decltype(auto)
      reduce(BinaryFunction&& b, Arg&& arg)
      {
        using P = std::decay_t<Nested>;
        return library_interface<P>::template reduce<indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg));
      }


      // to_stat_space not included

      // from_stat_space not included

      // wrap_angles not included


      template<typename Arg>
      static constexpr decltype(auto)
      conjugate(Arg&& arg)
      {
        if constexpr(has_untyped_index<Nested, 0>)
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
        if constexpr(has_untyped_index<Nested, 0>)
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
        if constexpr(has_untyped_index<Nested, 0>)
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
        if constexpr(has_untyped_index<Nested, 0>)
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
