/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief FromEuclideanExpr and related definitions.
 */

#ifndef OPENKALMAN_FROMEUCLIDEANEXPR_HPP
#define OPENKALMAN_FROMEUCLIDEANEXPR_HPP

#include "basics/traits/traits.hpp"

namespace OpenKalman
{

#ifdef __cpp_concepts
  template<has_untyped_index<0> NestedObject, patterns::pattern V0>
#else
  template<typename NestedObject, typename V0>
#endif
  struct FromEuclideanExpr : internal::AdapterBase<FromEuclideanExpr<NestedObject, V0>, NestedObject>
  {

  private:

#ifndef __cpp_concepts
    static_assert(indexible<NestedObject>);
    static_assert(patterns::pattern<V0>);
    static_assert(has_untyped_index<NestedObject, 0>);
#endif

    using Scalar = scalar_type_of_t<NestedObject>;

    using Base = internal::AdapterBase<FromEuclideanExpr, NestedObject>;

  public:

    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr FromEuclideanExpr() requires std::default_initializable<Base> and fixed_pattern<V0>
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdex::default_initializable<Base> and fixed_pattern<V0>, int> = 0>
    constexpr FromEuclideanExpr()
#endif
    {}


    /**
     * Construct from a compatible \ref indexible object.
     */
#ifdef __cpp_concepts
    template<indexible Arg, patterns::pattern D0> requires
      std::constructible_from<NestedObject, Arg&&> and std::constructible_from<std::decay_t<V0>, D0>
#else
    template<typename Arg, typename D0, std::enable_if_t<indexible<Arg> and patterns::pattern<C> and
      stdex::constructible_from<NestedObject, Arg&&> and stdex::constructible_from<std::decay_t<V0>, D0>, int> = 0>
#endif
    explicit FromEuclideanExpr(Arg&& arg, const D0& d0) : Base {std::forward<Arg>(arg)}, vector_space_descriptor_index_0{d0} {}


    /**
     * Construct from a compatible \ref indexible object if the \ref patterns::pattern "pattern" of index 0 is fixed.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires std::constructible_from<NestedObject, Arg&&> and fixed_index_descriptor<V0>
#else
    template<typename Arg, typename D0, std::enable_if_t<indexible<Arg> and
      stdex::constructible_from<NestedObject, Arg&&> and fixed_index_descriptor<V0>, int> = 0>
#endif
    explicit FromEuclideanExpr(Arg&& arg) : Base {std::forward<Arg>(arg)} {}


    /**
     * Assign from a compatible \ref indexible object.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires (not std::is_base_of_v<FromEuclideanExpr, std::decay_t<Arg>>) and
      std::assignable_from<std::add_lvalue_reference_t<NestedObject>, decltype(to_euclidean(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and (not std::is_base_of_v<ToEuclideanExpr, std::decay_t<Arg>>) and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedObject>, decltype(to_euclidean(std::declval<Arg&&>()))>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      using TArg = decltype(to_euclidean(std::declval<Arg>()));
      if constexpr ((zero<NestedObject> and zero<TArg>) or (identity_matrix<NestedObject> and identity_matrix<TArg>))
      {}
      else
      {
        this->nested_object() = to_euclidean(std::forward<Arg>(arg));
      }
      return *this;
    }

  protected:

    std::decay_t<V0> vector_space_descriptor_index_0;

    friend struct interface::object_traits<VectorSpaceAdapter>;
    friend struct interface::library_interface<VectorSpaceAdapter>;

  }; // struct FromEuclideanExpr


  // ------------------------------ //
  //        Deduction Guide         //
  // ------------------------------ //

#ifdef __cpp_concepts
  template<indexible Arg, patterns::pattern V>
#else
  template<typename Arg, typename V, std::enable_if_t<indexible<Arg> and patterns::pattern<V>, int> = 0>
#endif
  FromEuclideanExpr(Arg&&, const V&) -> FromEuclideanExpr<Arg, V>;


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {

    // --------------------------- //
    //   object_traits   //
    // --------------------------- //

    template<typename NestedObject, typename V0>
    struct object_traits<FromEuclideanExpr<NestedObject, V0>>
    {
      static const bool is_specialized = true;

      using scalar_type = scalar_type_of_t<NestedObject>;


      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg) { return OpenKalman::count_indices(nested_object(arg)); }


      template<typename Arg, typename N>
      static constexpr auto
      get_pattern_collection(Arg&& arg, const N& n)
      {
        if constexpr (values::fixed<N>)
        {
          if constexpr (n == 0_uz) return std::forward<Arg>(arg).vector_space_descriptor_index_0;
          else return OpenKalman::get_pattern_collection(nested_object(std::forward<Arg>(arg)), n);
        }
        else
        {
          using Desc = DynamicDescriptor<scalar_type_of<Arg>>;
          if (n == 0) return Desc {std::forward<Arg>(arg).vector_space_descriptor_index_0};
          else return Desc {OpenKalman::get_pattern_collection(nested_object(std::forward<Arg>(arg)), n)};
        }
      }


      template<typename Arg>
      static decltype(auto)
      nested_object(Arg&& arg)
      {
        return std::forward<Arg>(arg).nested_object();
      }


      template<typename Arg>
      static constexpr auto
      get_constant(const Arg& arg)
      {
        if constexpr (patterns::euclidean_pattern<V0>)
          return constant_value {arg.nested_object()};
        else
          return std::monostate {};
      }


      template<typename Arg>
      static constexpr auto
      get_constant_diagonal(const Arg& arg)
      {
        if constexpr (patterns::euclidean_pattern<V0>)
          return constant_diagonal_value {arg.nested_object()};
        else
          return std::monostate {};
      }


      template<applicability b>
      static constexpr bool
      one_dimensional = patterns::euclidean_pattern<V0> and OpenKalman::one_dimensional<NestedObject, b>;


      template<applicability b>
      static constexpr bool
      is_square = patterns::euclidean_pattern<V0> and square_shaped<NestedObject, b>;


      template<triangle_type t>
      static constexpr bool
      triangle_type_value = patterns::euclidean_pattern<V0> and triangular_matrix<NestedObject, t>;


      static constexpr bool
      is_triangular_adapter = false;


      static constexpr bool
      is_hermitian = patterns::euclidean_pattern<V0> and hermitian_matrix<NestedObject>;


    // hermitian_adapter_type is omitted


      static constexpr bool is_writable = false;


#ifdef __cpp_lib_concepts
      template<typename Arg> requires patterns::euclidean_pattern<V0> and raw_data_defined_for<nested_object_of_t<Arg&>>
#else
      template<typename Arg, std::enable_if_t<patterns::euclidean_pattern<V0> and raw_data_defined_for<typename nested_object_of<Arg&>::type>, int> = 0>
#endif
      static constexpr auto * const
      raw_data(Arg& arg)
      {
        return internal::raw_data(nested_object(arg));
      }


      static constexpr data_layout
      layout = patterns::euclidean_pattern<V0> ? layout_of_v<NestedObject> : data_layout::none;


#ifdef __cpp_concepts
      template<typename Arg> requires (layout != data_layout::none)
#else
      template<data_layout l = layout, typename Arg, std::enable_if_t<l != data_layout::none, int> = 0>
#endif
      static auto
      strides(Arg&& arg)
      {
        return OpenKalman::internal::strides(OpenKalman::nested_object(std::forward<Arg>(arg)));
      }

    };


    // --------------------- //
    //   library_interface   //
    // --------------------- //

    template<typename NestedObject, typename V0>
    struct library_interface<FromEuclideanExpr<NestedObject, V0>>
    {
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
      to_euclidean(Arg&& arg)
      {
        return nested_object(std::forward<Arg>(arg)); //< from- and then to- is an identity.
      }


      // from_euclidean not included. Double application of from_euclidean does not make sense.


      template<typename Arg>
      constexpr decltype(auto)
      wrap_angles(Arg&& arg)
      {
        return std::forward<Arg>(arg); //< A FromEuclideanExpr is already wrapped
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
      adjoint(Arg&& arg)
      {
        if constexpr(has_untyped_index<Arg, 0>)
        {
          return OpenKalman::adjoint(nested_object(std::forward<Arg>(arg)));
        }
        else
        {
          return std::forward<Arg>(arg).adjoint(); //< \todo Generalize this.
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


      template<HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
      static decltype(auto)
      rank_update_hermitian(A&& a, U&& u, const Alpha alpha)
      {
        return OpenKalman::rank_update_hermitian<significant_triangle>(make_hermitian_matrix(to_dense_object(std::forward<A>(a))), std::forward<U>(u), alpha);
      }


      template<triangle_type triangle, typename A, typename U, typename Alpha>
      static decltype(auto) rank_update_triangular(A&& a, U&& u, const Alpha alpha)
      {
        return OpenKalman::rank_update_triangular(make_triangular_matrix<triangle>(to_dense_object(std::forward<A>(a))), std::forward<U>(u), alpha);
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

    };


  }


} // OpenKalman


#endif
