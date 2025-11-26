/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for constant_adapter
 */

#ifndef OPENKALMAN_CONSTANT_ADAPTER_HPP
#define OPENKALMAN_CONSTANT_ADAPTER_HPP

#include "coordinates/coordinates.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/concepts/constant_object.hpp"
#include "linear-algebra/concepts/index_collection_for.hpp"
#include "linear-algebra/traits/internal/library_base.hpp"
#include "linear-algebra/traits/element_type_of.hpp"
#include "linear-algebra/traits/constant_value.hpp"
#include "linear-algebra/traits/get_pattern_collection.hpp"

namespace OpenKalman
{
  /**
   * \brief A tensor or other matrix in which all elements are a constant value.
   * \details The constant value can be any \ref values::value.
   * Examples:
   * \code
   * using T = Eigen::Matrix<double, 3, 2>; // A 3-by-2 matrix of scalar-type double in the Eigen library.
   * constant_adapter<double, T> c1 {3.0}; // Construct a 3-by-2 double constant within the library of T with value 3.0 (known at runtime).
   * constant_adapter<int, T> c2 {3}; // Construct a 3-by-2 int constant within the library of T with value 3 (known at runtime).
   * constant_adapter<value::fixed_value<int, 1>, T> c3; // Construct a 3-by-2 int constant within the library of T with value 1 (known at compile time).
   * constant_adapter<value::fixed_value<double, 1, T> c4; // Construct a 3-by-2 double constant within the library of T with value 1.0 (known at compile time).
   * constant_adapter<std::integral_constant<int, 1>, T> c5; // Construct a 3-by-2 int constant within the library of T with value 1 (known at compile time).
   * constant_adapter<value::fixed_value<std::complex<double>, T> c6 {std::complex<double>{4, 5}}; // Construct a 3-by-2 complex constant within the library of T and value 4.0 + 5.0i (known at runtime).
   * constant_adapter<value::fixed_value<std::complex<double, 4, 5>, T> c7; // Construct a 3-by-2 A complex constant within the library of T and value 4.0 + 5.0i (known at compile time).
   * \endcode
   * \tparam Value A \ref values::value.
   * \tparam Shape An \ref indexible object reflecting the size and shape of the adapter object.
   */
#ifdef __cpp_concepts
  template<values::value Value, indexible Shape> requires std::is_object_v<Value>
#else
  template<typename Value, typename Shape>
#endif
  struct constant_adapter : internal::library_base_t<constant_adapter<Value, Shape>, Shape>
  {
  private:
  
    using Pattern = decltype(get_pattern_collection(std::declval<const Shape&>()));
  
  public:
  
    /**
     * \brief Construct from \ref values::value "value" and a \ref coordinates::pattern_collection "pattern_collection"
     */
#ifdef __cpp_lib_ranges
    template<values::value V, coordinates::pattern_collection P> requires
      std::constructible_from<Value, V&&> and
      std::constructible_from<Pattern, P&&>
#else
    template<typename V, typename P, std::enable_if_t<
      values::value<V> and pattern_collection<P> and
      stdex::constructible_from<Value, V&&> and stdex::constructible_from<Pattern, P&&>, int> = 0>
#endif
    explicit constexpr constant_adapter(V&& v, P&& p)
      : value_ {std::forward<V>(v)}, pattern_ {std::forward<P>(p)} {}

  
    /**
     * \brief Construct from \ref values::value "value" and a reference to an \ref indexible object
     */
#ifdef __cpp_lib_ranges
    template<values::value V, indexible S> requires
      std::constructible_from<Value, V&&> and
      std::constructible_from<Pattern, decltype(get_pattern_collection(std::declval<S&&>()))>
#else
    template<typename V, typename S, std::enable_if_t<values::value<V> and
      stdex::constructible_from<Value, V&&> and
      stdex::constructible_from<Pattern, decltype(get_pattern_collection(std::declval<S&&>()))>, int> = 0>
#endif
    explicit constexpr constant_adapter(V&& v, S&& s)
      : value_ {std::forward<V>(v)}, pattern_ {get_pattern_collection(std::forward<S>(s))} {}


    /**
     * \overload
     * \brief Same as above, if the \ref pattern_collection associated with Shape is known at compile time.
     */
#ifdef __cpp_lib_ranges
    template<values::value V> requires
      std::constructible_from<Value, V&&> and
      coordinates::fixed_pattern_collection<Pattern>
#else
    template<typename V, typename S, std::enable_if_t<values::value<V> and
      stdex::constructible_from<Value, V&&> and coordinates::fixed_pattern_collection<Pattern>, int> = 0>
#endif
    explicit constexpr constant_adapter(V&& v)
      : value_ {std::forward<V>(v)}, pattern_ {} {}


    /**
     * \overload
     * \brief Default constructor, assuming the constant and the shape are known at compile time.
     */
#ifdef __cpp_lib_ranges
    explicit constexpr constant_adapter()
    requires values::fixed<Value> and coordinates::fixed_pattern_collection<Pattern>
#else
    template<bool Enable, std::enable_if_t<Enable and values::fixed<Value> and coordinates::fixed_pattern_collection<Pattern>, int> = 0>
    explicit constexpr constant_adapter()
#endif
      : value_ {}, pattern_ {} {}


    /**
     * \brief Construct from another \ref constant_object.
     */
#ifdef __cpp_concepts
    template<constant_object Arg> requires
      (not std::same_as<std::decay_t<Arg>, constant_adapter>) and
      std::constructible_from<Value, constant_value<Arg>> and
      std::constructible_from<Pattern, decltype(get_pattern_collection(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<constant_object<Arg> and
      (not stdex::same_as<std::decay_t<Arg>, constant_adapter>) and
      stdex::constructible_from<Value, constant_value<Arg>> and
      stdex::constructible_from<Pattern, decltype(get_pattern_collection(std::declval<Arg&&>()))>, int> = 0>
#endif
    constexpr constant_adapter(Arg&& arg) :
      value_ {constant_value {arg}},
      pattern_ {get_pattern_collection(std::forward<Arg>(arg))} {}


    /**
     * \brief Assign from a compatible \ref constant_object.
     */
#ifdef __cpp_concepts
    template<constant_object Arg> requires
      (not std::same_as<std::decay_t<Arg>, constant_adapter>) and
      std::assignable_from<Value&, constant_value<Arg>> and
      std::assignable_from<Pattern&, decltype(get_pattern_collection(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<constant_object<Arg> and
      (not stdex::same_as<std::decay_t<Arg>, constant_adapter>) and
      stdex::assignable_from<Value&, constant_value<Arg>> and
      stdex::assignable_from<Pattern&, decltype(get_pattern_collection(std::declval<Arg&&>()))>, int> = 0>
#endif
    constexpr auto& operator=(const Arg& arg)
    {
      value_ = constant_value {arg};
      pattern_ = get_pattern_collection(std::forward<Arg>(arg));
      return *this;
    }


    /**
     * \brief Access a component at a set of indices.
     * \return The element corresponding to the indices (always the constant).
     */
#ifdef __cpp_lib_ranges
    template<index_collection_for<Shape> Indices> requires (not empty_object<PatternMatrix>)
    constexpr values::scalar auto
#else
    template<typename Indices, std::enable_if_t<
      index_collection_for<Indices, Shape> and (not empty_object<PatternMatrix>), int> = 0>
    constexpr auto
#endif
    operator[](const Indices& indices) const 
    {
      return values::to_value_type(value_);
    }


    /**
     * \brief Get the \ref values::scalar associated with this object.
     */
#ifdef __cpp_concepts
    constexpr values::value auto
#else
    constexpr auto
#endif
    value() const
    {
      return value_;
    }

  protected:

    Value value_;

    Pattern pattern_;

    friend struct interface::object_traits<constant_adapter>;
    friend struct interface::library_interface<constant_adapter>;

  };


  // ------------------ //
  //  Deduction guides  //
  // ------------------ //

#ifdef __cpp_concepts
  template<values::value C, indexible Arg>
#else
  template<typename C, typename Arg, std::enable_if_t<values::scalar<C> and indexible<Arg>, int> = 0>
#endif
  constant_adapter(const C&, const Arg&) -> constant_adapter<C, Arg>;


#ifdef __cpp_concepts
  template<constant_object Arg> requires (not is_constant_adapter<Arg>::value)
#else
  template<typename Arg, std::enable_if_t<constant_object<Arg> and (not is_constant_adapter<Arg>), int> = 0>
#endif
  constant_adapter(const Arg&) -> constant_adapter<is_constant_value<Arg>::value, Arg>;


  // -------------- //
  //  zero_adapter  //
  // -------------- //

  /**
  * \brief A constant_adapter in which all elements are 0.
  * \tparam Shape An \ref indexible object reflecting the size and shape of the adapter object.
  * \tparam N A \ref values::number type (by default, derived from Shape).
  */
#ifdef __cpp_concepts
  template<indexible Shape, values::number N = element_type_of_t<Shape>>
#else
  template<typename Shape, typename N = element_type_of_t<Shape>>
#endif
  using zero_adapter = constant_adapter<values::fixed_value<N, 0>, Shape>;


  // ------------ //
  //  Interfaces  //
  // ------------ //

  namespace interface
  {
    template<typename PatternMatrix, typename Scalar, auto...constant>
    struct object_traits<constant_adapter<PatternMatrix, Scalar, constant...>>
    {
    private:

      using XprType = constant_adapter<PatternMatrix, Scalar, constant...>;

    public:

      using scalar_type = typename XprType::MyScalarType;
      using MyDims = typename XprType::MyDimensions_t;


      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg)
      {
        if constexpr (index_count_v<PatternMatrix> == stdex::dynamic_extent)
          return std::forward<Arg>(arg).pattern_.size();
        else
          return collections::size_of<MyDims> {};
      }


      template<typename Arg, typename N>
      static constexpr auto get_pattern_collection(Arg&& arg, const N& n)
      {
        if constexpr (index_count_v<PatternMatrix> == stdex::dynamic_extent)
        {
          return std::forward<Arg>(arg).pattern_[static_cast<typename MyDims::size_type>(n)];
        }
        else if constexpr (values::fixed<N>)
        {
          if constexpr (N::value >= index_count_v<PatternMatrix>) return Dimensions<1>{};
          else return std::get<N::value>(std::forward<Arg>(arg).pattern_);
        }
        else if (n >= collections::size_of_v<MyDims>)
        {
          return 1_uz;
        }
        else
        {
          return std::apply(
            [](auto&&...ds){ return std::array<std::size_t, collections::size_of_v<MyDims>> {std::forward<decltype(ds)>(ds)...}; },
            std::forward<Arg>(arg).pattern_)[n];
        }
      }


      // No nested_object defined


      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg) { return arg.value(); }


      // No get_constant_diagonal defined


      template<applicability b>
      static constexpr bool one_dimensional = OpenKalman::one_dimensional<PatternMatrix, b>;


      template<applicability b>
      static constexpr bool is_square = OpenKalman::square_shaped<PatternMatrix, b>;


      // No triangle_type_value, is_triangular_adapter, is_hermitian, or hermitian_adapter_type defined


      static constexpr bool is_writable = false;


      // No raw_data, layout, or strides defined.

    };


    template<typename PatternMatrix, typename Scalar, auto...constant>
    struct library_interface<constant_adapter<PatternMatrix, Scalar, constant...>>
    {
      template<typename Derived>
      using library_base = internal::library_base_t<Derived, PatternMatrix>;


      template<typename Arg, typename Indices>
      static constexpr auto
      access(Arg&& arg, const Indices&) { return std::forward<Arg>(arg).value(); }


      // No set_component defined  because constant_adapter is not writable.


      template<typename Arg>
      static decltype(auto)
      to_native_matrix(Arg&& arg)
      {
        return OpenKalman::to_native_matrix<PatternMatrix>(std::forward<Arg>(arg));
      }


      template<data_layout layout, typename S, typename D>
      static auto
      make_default(D&& d)
      {
        return make_dense_object<PatternMatrix, layout, S>(std::forward<D>(d));
      }


      // fill_components not necessary because T is not a dense writable matrix.


      template<typename C, typename D>
      static constexpr auto
      make_constant(C&& c, D&& d)
      {
        return OpenKalman::make_constant<PatternMatrix>(std::forward<C>(c), std::forward<D>(d));
      }


      template<typename S, typename D>
      static constexpr auto
      make_identity_matrix(D&& d)
      {
        return make_identity_matrix_like<PatternMatrix, S>(std::forward<D>(d));
      }


      // no get_slice
      // no set_slice
      // no set_triangle
      // no to_diagonal
      // no diagonal_of


      template<typename...Ds, typename Arg>
      static decltype(auto)
      replicate(const std::tuple<Ds...>& tup, Arg&& arg)
      {
        return library_interface<PatternMatrix>::replicate(tup, std::forward<Arg>(arg));
      }


      template<typename...Ds, typename Op, typename...Args>
      static constexpr decltype(auto)
      n_ary_operation(const std::tuple<Ds...>& d_tup, Op&& op, Args&&...args)
      {
        return library_interface<PatternMatrix>::n_ary_operation(d_tup, std::forward<Op>(op), std::forward<Args>(args)...);
      }


      template<std::size_t...indices, typename BinaryFunction, typename Arg>
      static constexpr decltype(auto)
      reduce(BinaryFunction&& b, Arg&& arg)
      {
        return library_interface<PatternMatrix>::template reduce<indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg));
      }


      // no to_euclidean
      // no from_euclidean
      // no wrap_angles

      // conjugate is not necessary because it is handled by the general conjugate function.
      // transpose is not necessary because it is handled by the general transpose function.
      // adjoint is not necessary because it is handled by the general adjoint function.
      // determinant is not necessary because it is handled by the general determinant function.


      template<typename A, typename B>
      static constexpr auto sum(A&& a, B&& b)
      {
        return library_interface<PatternMatrix>::sum(std::forward<A>(a), std::forward<B>(b));
      }


      template<typename A, typename B>
      static constexpr auto contract(A&& a, B&& b)
      {
        return library_interface<PatternMatrix>::contract(std::forward<A>(a), std::forward<B>(b));
      }


      // contract_in_place is not necessary because the argument will not be writable.

      // cholesky_factor is not necessary because it is handled by the general cholesky_factor function.


      template<HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
      static decltype(auto) rank_update_hermitian(A&& a, U&& u, const Alpha alpha)
      {
        using Trait = interface::library_interface<PatternMatrix>;
        return Trait::template rank_update_hermitian<significant_triangle>(std::forward<A>(a), std::forward<U>(u), alpha);
      }


      // rank_update_triangular is not necessary because it is handled by the general rank_update_triangular function.

      // solve is not necessary because it is handled by the general solve function.

      // LQ_decomposition is not necessary because it is handled by the general LQ_decomposition function.

      // QR_decomposition is not necessary because it is handled by the general QR_decomposition function.

    };

  }


}


#endif
